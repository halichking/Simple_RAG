"""Excel/CSV 数据分析与 Markdown 报表生成服务。"""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import ConfigData as info


XML_NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_index(cell_ref: str) -> int:
    """把 Excel 单元格引用里的列字母转换成从 0 开始的列下标。"""
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def _cell_position(cell_ref: str) -> tuple[int, int]:
    """把 A1 这类单元格引用转换成从 0 开始的 (行号, 列号)。"""
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    digits = "".join(ch for ch in cell_ref if ch.isdigit())
    row_index = int(digits or "1") - 1
    col_index = _column_index(letters)
    return row_index, col_index


def _merged_ranges(sheet_xml: ET.Element) -> list[tuple[int, int, int, int]]:
    """
    读取当前 sheet 的合并单元格范围。

    xlsx 的合并单元格底层只有左上角单元格保存真实值，其他格天然为空。
    如果不先展开合并区域，后续统计会把这些视觉上有值的格子误判成空白/未填写。
    """
    ranges: list[tuple[int, int, int, int]] = []
    for merge_node in sheet_xml.findall(".//main:mergeCell", XML_NS):
        ref = merge_node.attrib.get("ref", "")
        if ":" not in ref:
            continue
        start_ref, end_ref = ref.split(":", 1)
        start_row, start_col = _cell_position(start_ref)
        end_row, end_col = _cell_position(end_ref)
        ranges.append((start_row, start_col, end_row, end_col))
    return ranges


def _expand_merged_cells(rows: list[list[str]], ranges: list[tuple[int, int, int, int]]) -> list[list[str]]:
    """
    将合并单元格按左上角值展开到整个合并区域。

    这是 xlsx 读取阶段的通用处理，适用于所有用户上传的 xlsx。
    """
    if not ranges:
        return rows

    expanded_rows = [list(row) for row in rows]
    for start_row, start_col, end_row, end_col in ranges:
        # 横向合并通常是报表大标题。
        # 这种标题不能展开成每一列的值，否则表头识别会把所有字段都误读成标题文本。
        # 只有跨行合并才更像“层级主字段沿用”，适合展开到数据区。
        if start_row == end_row:
            continue
        if start_row >= len(expanded_rows):
            continue
        if start_col >= len(expanded_rows[start_row]):
            continue
        fill_value = expanded_rows[start_row][start_col]
        if not fill_value:
            continue

        for row_index in range(start_row, end_row + 1):
            while len(expanded_rows) <= row_index:
                expanded_rows.append([])
            while len(expanded_rows[row_index]) <= end_col:
                expanded_rows[row_index].append("")
            for col_index in range(start_col, end_col + 1):
                if not expanded_rows[row_index][col_index]:
                    expanded_rows[row_index][col_index] = fill_value
    return expanded_rows


def _safe_text(value: Any) -> str:
    """把单元格值统一转成去掉首尾空白的字符串，方便后续统计。"""
    if value is None:
        return ""
    return str(value).strip()


def _file_md5(path: Path) -> str:
    """计算文件 MD5，用于跳过完全重复的 Excel 导出文件。"""
    md5_obj = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            md5_obj.update(chunk)
    return md5_obj.hexdigest()


def _number(value: Any, default: float = 0.0) -> float:
    """把 Excel 中可能出现的数字字符串转换成 float，失败时返回默认值。"""
    try:
        if value in ("", None):
            return default
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return default


def _ratio(done: Any, total: Any) -> float:
    """计算完成率，避免除零和脏数据导致报表生成失败。"""
    total_number = _number(total)
    if total_number <= 0:
        return 0.0
    return _number(done) / total_number


def _top_items(counter: Counter, limit: int = 5) -> list[tuple[str, int]]:
    """提取 Top N，过滤空值，保证报表里不会出现大量无意义空分类。"""
    return [(name, count) for name, count in counter.most_common(limit) if name]


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """生成简单 Markdown 表格，供网页端直接渲染或复制。"""
    if not rows:
        return "暂无数据"
    header_line = "| " + " | ".join(headers) + " |"
    split_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(_safe_text(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, split_line] + body_lines)


def _read_shared_strings(zip_file: zipfile.ZipFile) -> list[str]:
    """读取 xlsx 共享字符串表；很多系统导出的中文表头会存放在这里。"""
    try:
        raw_xml = zip_file.read("xl/sharedStrings.xml")
    except KeyError:
        return []

    root = ET.fromstring(raw_xml)
    strings: list[str] = []
    for item in root.findall("main:si", XML_NS):
        parts = [text_node.text or "" for text_node in item.findall(".//main:t", XML_NS)]
        strings.append("".join(parts))
    return strings


def _read_workbook_sheet_paths(zip_file: zipfile.ZipFile) -> list[str]:
    """读取工作簿里的 sheet 路径；当前导出文件通常只有一个 sheet，但这里保留多 sheet 支持。"""
    workbook_xml = ET.fromstring(zip_file.read("xl/workbook.xml"))
    rels_xml = ET.fromstring(zip_file.read("xl/_rels/workbook.xml.rels"))

    rel_map: dict[str, str] = {}
    for rel in rels_xml:
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target", "")
        if rel_id and target:
            rel_map[rel_id] = "xl/" + target.lstrip("/")

    sheet_paths: list[str] = []
    for sheet in workbook_xml.findall(".//main:sheet", XML_NS):
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if rel_id and rel_id in rel_map:
            sheet_paths.append(rel_map[rel_id])
    return sheet_paths


def read_xlsx_rows(path: Path) -> list[list[str]]:
    """
    用标准库读取 xlsx 的所有行。

    这里没有使用 pandas/openpyxl，是为了让项目在当前虚拟环境中直接可运行。
    该读取器覆盖本项目导出的普通文本/数字表格场景，不处理复杂公式和样式。
    """
    with zipfile.ZipFile(path) as zip_file:
        shared_strings = _read_shared_strings(zip_file)
        sheet_paths = _read_workbook_sheet_paths(zip_file)
        all_rows: list[list[str]] = []

        for sheet_path in sheet_paths:
            sheet_xml = ET.fromstring(zip_file.read(sheet_path))
            sheet_rows: list[list[str]] = []
            for row_node in sheet_xml.findall(".//main:row", XML_NS):
                row_values: list[str] = []
                for cell_node in row_node.findall("main:c", XML_NS):
                    cell_ref = cell_node.attrib.get("r", "")
                    col_index = _column_index(cell_ref) if cell_ref else len(row_values)
                    while len(row_values) <= col_index:
                        row_values.append("")

                    cell_type = cell_node.attrib.get("t")
                    value_node = cell_node.find("main:v", XML_NS)
                    inline_node = cell_node.find("main:is/main:t", XML_NS)

                    if cell_type == "s" and value_node is not None:
                        value_index = int(value_node.text or 0)
                        value = shared_strings[value_index] if value_index < len(shared_strings) else ""
                    elif cell_type == "inlineStr" and inline_node is not None:
                        value = inline_node.text or ""
                    elif value_node is not None:
                        value = value_node.text or ""
                    else:
                        value = ""

                    row_values[col_index] = _safe_text(value)
                sheet_rows.append(row_values)

            # 通用展开合并单元格：把视觉上合并显示的值写回每个底层空格，
            # 防止后续“空白/未填写”统计把合并区域误认为缺数据。
            all_rows.extend(_expand_merged_cells(sheet_rows, _merged_ranges(sheet_xml)))
        return all_rows


@dataclass
class ExcelSource:
    """记录每个 Excel 文件的读取结果和去重信息，方便健康检查与报表溯源。"""

    path: Path
    md5: str
    rows: list[list[str]]
    skipped_duplicate: bool = False


class ReportService:
    """负责读取 Excel/CSV 表格，并生成 Markdown 分析摘要。"""

    def __init__(self, docs_dir: Path | None = None):
        self.docs_dir = docs_dir or info.docs_dir
        self.duplicate_strategy = info.report_duplicate_strategy

    def _load_excel_sources(self, pattern: str) -> list[ExcelSource]:
        """按文件名模式读取 Excel，并按 MD5 跳过完全重复的导出文件。"""
        sources: list[ExcelSource] = []
        seen_md5: set[str] = set()
        for path in sorted(self.docs_dir.glob(pattern)):
            if path.name.startswith("~$"):
                continue
            digest = _file_md5(path)
            skipped = self.duplicate_strategy == "md5" and digest in seen_md5
            if not skipped:
                seen_md5.add(digest)
            rows = [] if skipped else read_xlsx_rows(path)
            sources.append(ExcelSource(path=path, md5=digest, rows=rows, skipped_duplicate=skipped))
        return sources

    @staticmethod
    def _rows_to_dicts(rows: list[list[str]], header_row_index: int = 1) -> list[dict[str, str]]:
        """把第二行表头之后的数据行转成字典列表，自动补齐缺失列。"""
        if len(rows) <= header_row_index:
            return []
        headers = [_safe_text(header) for header in rows[header_row_index]]
        records: list[dict[str, str]] = []
        for raw_row in rows[header_row_index + 1 :]:
            row = raw_row + [""] * max(0, len(headers) - len(raw_row))
            record = {header: _safe_text(row[index]) for index, header in enumerate(headers) if header}
            if any(record.values()):
                records.append(record)
        return records

    def generate_markdown_report(self, question: str | None = None) -> str:
        """生成 docs 目录中所有表格文件的通用 Markdown 摘要。"""
        file_paths = sorted(
            path for path in self.docs_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".xlsx", ".xls", ".csv"} and not path.name.startswith("~$")
        )
        if not file_paths:
            return (
                "# 表格数据总结报表\n\n"
                f"> 生成范围：`{self.docs_dir}`。"
                + (f" 用户问题：{question}" if question else "")
                + "\n\n当前目录没有可分析的 Excel/CSV 文件。"
            )
        return self.generate_uploaded_files_report(file_paths=file_paths, question=question)

    def generate_uploaded_files_report(self, file_paths: list[Path], question: str | None = None) -> str:
        """
        对聊天框临时上传的表格做通用统计摘要。

        这里仍由程序负责准确读数、计数、Top 项和空白统计；后续 RagService 会把这份
        稳定摘要交给大模型做自然语言解读，避免模型直接猜数。
        """
        if not file_paths:
            return "当前会话没有上传可分析的表格文件。"

        from session_file_service import read_table_rows

        lines = [
            "# 当前会话上传文件分析摘要",
            "",
            f"> 用户问题：{question or '未提供具体问题'}",
        ]

        for file_path in file_paths:
            rows = read_table_rows(file_path)
            table = self._analyze_generic_rows(rows)
            lines.extend(
                [
                    "",
                    f"## 文件：{file_path.name}",
                    "",
                    f"- 数据行数：{table['row_count']} 行。",
                    f"- 字段数量：{len(table['headers'])} 个。",
                    f"- 字段列表：{', '.join(table['headers'][:30]) or '未识别'}。",
                    "",
                    "### 空白/未填写较多字段",
                    "",
                    "> 说明：这里统计的是表格底层为空的单元格；xlsx 合并单元格已在读取阶段按左上角值自动展开，不会再被当成空白。",
                    _markdown_table(["字段", "空白数量"], table["missing_top"]),
                    "",
                    "### 主要分类字段 Top 项",
                    table["category_markdown"],
                    "",
                    "### 数值字段概览",
                    _markdown_table(["字段", "数量", "最小值", "最大值", "平均值"], table["numeric_rows"]),
                    "",
                    "### 样例数据",
                    _markdown_table(table["headers"][:8], [row[:8] for row in table["sample_rows"]]),
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _analyze_generic_rows(rows: list[list[str]]) -> dict[str, Any]:
        """对任意表格做通用统计，适配用户临时上传的未知结构 Excel/CSV。"""
        if not rows:
            return {
                "headers": [],
                "row_count": 0,
                "missing_top": [],
                "category_markdown": "暂无数据",
                "numeric_rows": [],
                "sample_rows": [],
            }

        header_index = 0
        headers: list[str] = []
        for index, row in enumerate(rows[:5]):
            cleaned = [_safe_text(cell) for cell in row]
            non_empty = [cell for cell in cleaned if cell]
            if len(non_empty) >= 2:
                header_index = index
                headers = cleaned
                break

        headers = [header or f"字段{index + 1}" for index, header in enumerate(headers)]
        data_rows = []
        for raw_row in rows[header_index + 1 :]:
            row = [_safe_text(cell) for cell in raw_row]
            row = row + [""] * max(0, len(headers) - len(row))
            if any(row):
                data_rows.append(row[: len(headers)])

        data_rows = ReportService._normalize_hierarchical_rows(headers, data_rows)

        missing_counter = Counter()
        column_values: dict[str, list[str]] = {header: [] for header in headers}
        for row in data_rows:
            for index, header in enumerate(headers):
                value = row[index] if index < len(row) else ""
                if not value:
                    missing_counter[header] += 1
                else:
                    column_values[header].append(value)

        # 整列都为空的字段通常是导出模板里的可选字段，例如“提交人/备注”。
        # 这类字段不代表业务数据缺失，避免在报告里误导用户。
        missing_top = [
            (name, count)
            for name, count in missing_counter.most_common(8)
            if column_values.get(name)
        ]

        category_sections: list[str] = []
        numeric_rows: list[list[Any]] = []
        for header, values in column_values.items():
            if not values:
                continue

            numeric_values: list[float] = []
            for value in values:
                try:
                    numeric_values.append(float(value.replace(",", "")))
                except ValueError:
                    pass

            if len(numeric_values) >= max(3, len(values) * 0.8):
                average = sum(numeric_values) / len(numeric_values)
                numeric_rows.append(
                    [
                        header,
                        len(numeric_values),
                        f"{min(numeric_values):.2f}",
                        f"{max(numeric_values):.2f}",
                        f"{average:.2f}",
                    ]
                )
                continue

            unique_count = len(set(values))
            if 1 < unique_count <= min(20, max(3, len(values) // 2)):
                top_rows = _top_items(Counter(values), 5)
                category_sections.append(f"**{header}**\n\n{_markdown_table(['取值', '数量'], top_rows)}")

        return {
            "headers": headers,
            "row_count": len(data_rows),
            "missing_top": missing_top,
            "category_markdown": "\n\n".join(category_sections[:6]) or "暂无明显分类字段",
            "numeric_rows": numeric_rows[:8],
            "sample_rows": data_rows[:5],
        }

    @staticmethod
    def _normalize_hierarchical_rows(headers: list[str], rows: list[list[str]]) -> list[list[str]]:
        """对常见“主记录 + 明细行”导出格式做向下填充。"""
        fill_fields = [field for field in headers if any(key in field for key in ["项目", "部门", "名称", "状态", "时间"])]
        if not fill_fields:
            return rows
        fill_indexes = [headers.index(field) for field in fill_fields if field in headers]
        current_values: dict[int, str] = {}
        normalized_rows: list[list[str]] = []

        for row in rows:
            normalized = list(row)
            for index in fill_indexes:
                if index < len(normalized) and normalized[index]:
                    current_values[index] = normalized[index]
                elif index in current_values:
                    normalized[index] = current_values[index]
            normalized_rows.append(normalized)
        return normalized_rows
