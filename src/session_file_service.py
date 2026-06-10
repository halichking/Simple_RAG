"""当前会话临时分析文件管理服务。"""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path
from typing import Any

import ConfigData as info
from report_service import read_xlsx_rows


ANALYSIS_SUFFIXES = {".xlsx", ".xls", ".csv"}


def safe_session_id(session_id: str | None) -> str:
    """把前端传来的 session_id 清洗成安全目录名，避免路径穿越。"""
    raw_value = session_id or info.default_session_id
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_value)
    return cleaned[:80] or info.default_session_id


def safe_filename(filename: str) -> str:
    """清洗上传文件名，只保留最后一级文件名和常见安全字符。"""
    raw_name = Path(filename or "upload").name
    cleaned = re.sub(r"[^a-zA-Z0-9_.\-\u4e00-\u9fff ()（）]", "_", raw_name)
    return cleaned[:120] or "upload"


def read_csv_rows(path: Path) -> list[list[str]]:
    """读取 CSV 文件，自动尝试 utf-8-sig 和 gbk 两种常见编码。"""
    for encoding in ("utf-8-sig", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                return [[str(cell).strip() for cell in row] for row in csv.reader(file)]
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as file:
        return [[str(cell).strip() for cell in row] for row in csv.reader(file)]


def read_table_rows(path: Path) -> list[list[str]]:
    """根据文件后缀读取表格行数据。"""
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return read_xlsx_rows(path)
    if suffix == ".csv":
        return read_csv_rows(path)
    if suffix == ".xls":
        try:
            import xlrd
        except Exception as exc:
            raise RuntimeError("当前环境缺少 xlrd，暂时无法读取 .xls 文件；请转存为 .xlsx 或 .csv 后上传") from exc

        workbook = xlrd.open_workbook(str(path))
        rows: list[list[str]] = []
        for sheet in workbook.sheets():
            for row_index in range(sheet.nrows):
                rows.append([str(sheet.cell_value(row_index, col)).strip() for col in range(sheet.ncols)])
        return rows
    raise ValueError(f"不支持的分析文件格式：{suffix}")


class SessionFileService:
    """管理当前会话的临时上传文件。"""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or info.session_upload_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str | None) -> Path:
        """返回当前会话的临时文件目录。"""
        directory = self.base_dir / safe_session_id(session_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def save_analysis_file(self, session_id: str | None, filename: str, content: bytes) -> dict[str, Any]:
        """保存聊天框 + 上传的临时分析文件，并返回文件摘要。"""
        clean_name = safe_filename(filename)
        suffix = Path(clean_name).suffix.lower()
        if suffix not in ANALYSIS_SUFFIXES:
            raise ValueError("聊天框 + 仅支持 .xlsx、.xls、.csv 临时分析文件")

        target = self.session_dir(session_id) / clean_name
        target.write_bytes(content)
        summary = self.summarize_file(target)
        summary["saved_path"] = str(target)
        return summary

    def list_files(self, session_id: str | None) -> list[Path]:
        """列出当前会话已上传的临时分析文件。"""
        directory = self.session_dir(session_id)
        return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in ANALYSIS_SUFFIXES)

    def resolve_files(self, session_id: str | None, file_names: list[str] | None) -> list[Path]:
        """
        根据“本次消息附件文件名”解析真实文件路径。

        这个方法故意不会在 file_names 为空时返回会话里的全部文件，因为聊天软件里的
        附件语义是“本条消息带了哪些文件”，不是“当前会话曾经上传过哪些文件”。这样可以
        避免用户第二次上传新表格时，第一次上传的旧表格又被自动拉进来重复分析。
        """
        if not file_names:
            return []

        directory = self.session_dir(session_id).resolve()
        available_files = {path.name: path for path in self.list_files(session_id)}
        resolved_files: list[Path] = []
        seen_names: set[str] = set()

        for raw_name in file_names:
            clean_name = safe_filename(str(raw_name))
            if clean_name in seen_names:
                continue
            seen_names.add(clean_name)

            candidate = available_files.get(clean_name)
            if candidate is None:
                raise FileNotFoundError(f"当前会话找不到临时分析文件：{clean_name}")

            # 双重确认最终路径仍在会话临时目录内，避免前端伪造文件名造成路径穿越。
            resolved = candidate.resolve()
            if directory not in resolved.parents:
                raise ValueError(f"非法临时文件路径：{clean_name}")
            resolved_files.append(resolved)

        return resolved_files

    def clear_session(self, session_id: str | None) -> int:
        """删除当前会话临时分析目录，返回删除文件数量。"""
        directory = self.base_dir / safe_session_id(session_id)
        if not directory.exists():
            return 0
        file_count = sum(1 for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory)
        return file_count

    def summarize_file(self, path: Path) -> dict[str, Any]:
        """读取上传表格的基础结构信息，用于上传成功后的前端提示。"""
        rows = read_table_rows(path)
        header_index, header = self._guess_header(rows)
        data_rows = max(0, len([row for row in rows[header_index + 1 :] if any(cell.strip() for cell in row)]))
        return {
            "file_name": path.name,
            "suffix": path.suffix.lower(),
            "row_count": data_rows,
            "column_count": len(header),
            "columns": header[:30],
        }

    def summarize_session(self, session_id: str | None) -> list[dict[str, Any]]:
        """返回当前会话所有临时分析文件摘要。"""
        summaries: list[dict[str, Any]] = []
        for path in self.list_files(session_id):
            try:
                summaries.append(self.summarize_file(path))
            except Exception as exc:
                summaries.append({"file_name": path.name, "error": str(exc)})
        return summaries

    @staticmethod
    def _guess_header(rows: list[list[str]]) -> tuple[int, list[str]]:
        """从表格前几行猜测表头，跳过系统导出的标题行。"""
        for index, row in enumerate(rows[:5]):
            cleaned = [cell.strip() for cell in row if cell.strip()]
            if len(cleaned) >= 2:
                return index, cleaned
        return 0, []
