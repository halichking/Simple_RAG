"""本地轻量网页小助手服务。"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = Path(__file__).resolve().parent
for module_path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

import ConfigData as info
from session_file_service import ANALYSIS_SUFFIXES, SessionFileService, safe_filename
from vector_retriever import VectorRetrieveService


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>本地知识库 RAG 助手</title>
  <style>
    :root {
      --bg: #f6f7fb;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --border: #d8dee9;
      --primary: #0f766e;
      --primary-dark: #115e59;
      --code: #f1f5f9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
    }
    .shell {
      width: min(1120px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.25;
    }
    .status {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .workspace {
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
    }
    .chat {
      min-height: 70vh;
      display: flex;
      flex-direction: column;
    }
    .messages {
      flex: 1;
      padding: 18px;
      overflow: auto;
    }
    .message {
      margin-bottom: 16px;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      line-height: 1.65;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .message h1,
    .message h2,
    .message h3 {
      margin: 12px 0 8px;
      line-height: 1.35;
    }
    .message h1 { font-size: 21px; }
    .message h2 { font-size: 18px; }
    .message h3 { font-size: 16px; }
    .message ul {
      margin: 8px 0;
      padding-left: 22px;
    }
    .message table {
      width: 100%;
      border-collapse: collapse;
      margin: 10px 0;
      font-size: 14px;
      white-space: normal;
    }
    .message th,
    .message td {
      border: 1px solid var(--border);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
    }
    .message th {
      background: #f8fafc;
      font-weight: 700;
    }
    .message.user {
      border-color: #99f6e4;
      background: #f0fdfa;
    }
    .message.thinking {
      color: var(--muted);
    }
    .thinking-line {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      min-height: 24px;
    }
    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid #cbd5e1;
      border-top-color: var(--primary);
      border-radius: 999px;
      animation: spin 0.85s linear infinite;
    }
    .thinking-dots::after {
      content: "";
      animation: dots 1.2s steps(4, end) infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
    @keyframes dots {
      0% { content: ""; }
      25% { content: "."; }
      50% { content: ".."; }
      75%, 100% { content: "..."; }
    }
    .role {
      display: block;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    form {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--border);
    }
    .file-tools {
      display: flex;
      align-items: flex-end;
    }
    .icon-button {
      width: 48px;
      min-width: 48px;
      height: 48px;
      font-size: 24px;
      line-height: 1;
      padding: 0;
    }
    .file-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 0 14px 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .composer-attachments {
      display: none;
      flex-wrap: wrap;
      gap: 10px;
      padding: 12px 14px 0;
      border-top: 1px solid var(--border);
      background: #fff;
    }
    .composer-attachments.active {
      display: flex;
    }
    .attachment-card {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      width: min(280px, 100%);
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #f8fafc;
      color: var(--text);
      font-size: 13px;
    }
    .attachment-name {
      font-weight: 700;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .attachment-meta {
      grid-column: 1 / -1;
      color: var(--muted);
      line-height: 1.4;
    }
    .attachment-remove {
      width: 24px;
      min-width: 24px;
      height: 24px;
      border-radius: 999px;
      background: #111827;
      color: #fff;
      font-size: 14px;
      line-height: 1;
    }
    .file-chip {
      padding: 5px 8px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #f8fafc;
    }
    textarea {
      width: 100%;
      min-height: 48px;
      max-height: 160px;
      resize: vertical;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font: inherit;
      line-height: 1.5;
    }
    button {
      min-width: 92px;
      border: 0;
      border-radius: 8px;
      background: var(--primary);
      color: #fff;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
      background: #94a3b8;
    }
    aside {
      padding: 16px;
    }
    aside h2 {
      margin: 0 0 12px;
      font-size: 16px;
    }
    .quick {
      display: grid;
      gap: 8px;
    }
    .quick button {
      min-width: 0;
      width: 100%;
      padding: 10px;
      background: #eef2ff;
      color: #312e81;
      text-align: left;
      font-weight: 600;
    }
    .quick button.danger {
      background: #fee2e2;
      color: #991b1b;
    }
    .quick button.secondary {
      background: #ecfeff;
      color: #155e75;
    }
    code {
      padding: 2px 5px;
      border-radius: 4px;
      background: var(--code);
    }
    @media (max-width: 840px) {
      header { align-items: flex-start; flex-direction: column; }
      .workspace { grid-template-columns: 1fr; }
      form { grid-template-columns: 1fr; }
      button { min-height: 44px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>本地知识库 RAG 助手</h1>
      </div>
      <div id="status" class="status">正在检查服务状态...</div>
    </header>
    <section class="workspace">
      <div class="panel chat">
        <div id="messages" class="messages">
          <div class="message">
            <span class="role">助手</span>
            你好，我可以回答本地知识库问题，也可以分析临时上传的 Excel/CSV 表格。
          </div>
        </div>
        <div id="composer-attachments" class="composer-attachments"></div>
        <form id="chat-form">
          <div class="file-tools">
            <button id="analysis-upload-button" class="icon-button" type="button" title="上传临时分析文件">+</button>
            <input id="analysis-file" type="file" accept=".xlsx,.xls,.csv" multiple hidden />
          </div>
          <textarea id="question" placeholder="输入问题，例如：分析附件这个表，或：这份文档的主要流程是什么？"></textarea>
          <button id="send" type="submit">发送</button>
        </form>
        <div id="file-strip" class="file-strip"></div>
      </div>
      <aside class="panel">
        <h2>快捷问题</h2>
        <div class="quick">
          <button type="button" id="knowledge-upload" class="secondary">上传资料到知识库</button>
          <input id="knowledge-file" type="file" accept=".pdf,.txt" hidden />
          <button type="button" id="sync-docs">同步 docs 知识库</button>
          <button type="button" data-question="生成当前表格数据总结报表">生成当前数据总结报表</button>
          <button type="button" data-question="分析状态和分类分布">分析状态和分类</button>
          <button type="button" data-question="分析任务完成情况">分析任务完成情况</button>
          <button type="button" data-question="这份文档的主要流程是什么？">知识库流程问答</button>
          <button type="button" id="clear-history" class="danger">清空当前会话历史</button>
          <button type="button" id="clear-knowledge" class="danger">清空知识库</button>
        </div>
      </aside>
    </section>
  </main>
  <script>
    const messages = document.querySelector("#messages");
    const form = document.querySelector("#chat-form");
    const question = document.querySelector("#question");
    const send = document.querySelector("#send");
    const statusBox = document.querySelector("#status");
    const fileStrip = document.querySelector("#file-strip");
    const composerAttachments = document.querySelector("#composer-attachments");
    const analysisFile = document.querySelector("#analysis-file");
    const knowledgeFile = document.querySelector("#knowledge-file");
    const sessionId = localStorage.getItem("rag_session_id") || "local_web_user";
    localStorage.setItem("rag_session_id", sessionId);
    // 当前输入框待发送附件：只跟下一条用户消息绑定，发送成功后清空。
    // 会话历史文件另由 /session-files 展示，不会自动进入本次分析。
    let pendingAnalysisFiles = [];

    function addMessage(role, content, className = "") {
      const item = document.createElement("div");
      item.className = `message ${className}`;
      item.innerHTML = `<span class="role">${role}</span>${renderMarkdown(content)}`;
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      return item;
    }

    function addThinkingMessage(role = "助手") {
      // 专门用于等待后端首个响应包的占位消息。
      // 收到真实内容后会被 ask() 替换成普通 Markdown，避免动画残留在最终回答里。
      const item = document.createElement("div");
      item.className = "message thinking";
      item.innerHTML = `
        <span class="role">${role}</span>
        <div class="thinking-line">
          <span class="spinner" aria-hidden="true"></span>
          <span><span class="thinking-dots">思考中</span></span>
        </div>
      `;
      messages.appendChild(item);
      messages.scrollTop = messages.scrollHeight;
      return item;
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderMarkdown(text) {
      const lines = String(text).split("\\n");
      const html = [];
      let inList = false;
      let inTable = false;

      function closeBlocks() {
        if (inList) {
          html.push("</ul>");
          inList = false;
        }
        if (inTable) {
          html.push("</tbody></table>");
          inTable = false;
        }
      }

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        const escaped = escapeHtml(line);
        if (!line.trim()) {
          closeBlocks();
          html.push("<br>");
          continue;
        }
        if (line.startsWith("|") && line.endsWith("|")) {
          const cells = line.slice(1, -1).split("|").map((cell) => escapeHtml(cell.trim()));
          const isDivider = cells.every((cell) => /^-+$/.test(cell.replaceAll(" ", "")));
          if (isDivider) continue;
          if (!inTable) {
            closeBlocks();
            html.push("<table><tbody>");
            inTable = true;
          }
          html.push(`<tr>${cells.map((cell) => `<td>${cell}</td>`).join("")}</tr>`);
          continue;
        }
        closeBlocks();
        if (line.startsWith("### ")) {
          html.push(`<h3>${escapeHtml(line.slice(4))}</h3>`);
        } else if (line.startsWith("## ")) {
          html.push(`<h2>${escapeHtml(line.slice(3))}</h2>`);
        } else if (line.startsWith("# ")) {
          html.push(`<h1>${escapeHtml(line.slice(2))}</h1>`);
        } else if (line.startsWith("- ")) {
          if (!inList) {
            html.push("<ul>");
            inList = true;
          }
          html.push(`<li>${escapeHtml(line.slice(2))}</li>`);
        } else {
          html.push(`<div>${escaped}</div>`);
        }
      }
      closeBlocks();
      return html.join("");
    }

    async function checkHealth() {
      try {
        const res = await fetch(`/health?session_id=${encodeURIComponent(sessionId)}`);
        const data = await res.json();
        statusBox.textContent = `知识库：${data.knowledge_count || 0} | 历史记忆：${data.history_count || 0} | 临时文件：${data.session_file_count || 0} | MongoDB：${data.mongodb ? "可用" : "未连接"}`;
      } catch (error) {
        statusBox.textContent = "服务状态检查失败";
      }
    }

    async function refreshSessionFiles() {
      try {
        const res = await fetch(`/session-files?session_id=${encodeURIComponent(sessionId)}`);
        const data = await res.json();
        const files = data.files || [];
        renderFileStrip(files);
      } catch (error) {
        renderFileStrip([]);
      }
    }

    function renderFileStrip(historyFiles = []) {
      const historyHtml = historyFiles.length
        ? historyFiles.map((file) => `<span class="file-chip">历史参考：${escapeHtml(file.file_name)} · ${file.row_count || 0}行</span>`).join("")
        : "";
      fileStrip.innerHTML = historyHtml || "";
      renderComposerAttachments();
    }

    function renderComposerAttachments() {
      if (!pendingAnalysisFiles.length) {
        composerAttachments.classList.remove("active");
        composerAttachments.innerHTML = "";
        return;
      }
      composerAttachments.classList.add("active");
      composerAttachments.innerHTML = pendingAnalysisFiles.map((file) => `
        <div class="attachment-card">
          <div class="attachment-name" title="${escapeHtml(file.file_name)}">${escapeHtml(file.file_name)}</div>
          <button class="attachment-remove" type="button" data-remove-file="${escapeHtml(file.file_name)}" title="移除待发送附件">×</button>
          <div class="attachment-meta">${file.row_count || 0} 行 · ${file.column_count || 0} 列</div>
        </div>
      `).join("");
      composerAttachments.querySelectorAll("[data-remove-file]").forEach((button) => {
        button.addEventListener("click", () => {
          const fileName = button.dataset.removeFile;
          pendingAnalysisFiles = pendingAnalysisFiles.filter((file) => file.file_name !== fileName);
          renderComposerAttachments();
        });
      });
    }

    async function ask(text) {
      const currentAttachments = pendingAnalysisFiles.map((file) => file.file_name);
      const trimmed = text.trim() || (currentAttachments.length ? "分析这些临时文件" : "");
      if (!trimmed) return;
      const userContent = currentAttachments.length
        ? `${trimmed}\n\n附件：${currentAttachments.join("、")}`
        : trimmed;
      addMessage("你", userContent, "user");
      question.value = "";
      send.disabled = true;
      const pending = addThinkingMessage("助手");
      try {
        const res = await fetch("/chat-stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: trimmed,
            session_id: sessionId,
            analysis_files: currentAttachments
          })
        });
        if (!res.ok || !res.body) {
          throw new Error(`HTTP ${res.status}`);
        }
        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let answer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\\n");
          buffer = lines.pop() || "";
          for (const line of lines) {
            if (!line.trim()) continue;
            const event = JSON.parse(line);
            if (event.type === "error") {
              throw new Error(event.delta || "服务处理失败");
            }
            if (event.type === "done") {
              continue;
            }
            answer += event.delta || "";
            pending.classList.remove("thinking");
            pending.innerHTML = `<span class="role">助手</span>${renderMarkdown(answer)}`;
            messages.scrollTop = messages.scrollHeight;
          }
        }
        if (!answer) {
          pending.classList.remove("thinking");
          pending.innerHTML = `<span class="role">助手</span>${renderMarkdown("已完成，但没有返回内容。")}`;
        }
        if (res.ok) {
          pendingAnalysisFiles = [];
          renderComposerAttachments();
          await refreshSessionFiles();
        }
      } catch (error) {
        pending.innerHTML = `<span class="role">助手</span>请求失败：${escapeHtml(error.message)}`;
      } finally {
        send.disabled = false;
        question.focus();
      }
    }

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      ask(question.value);
    });

    document.querySelectorAll("[data-question]").forEach((button) => {
      button.addEventListener("click", () => ask(button.dataset.question));
    });

    document.querySelector("#sync-docs").addEventListener("click", async () => {
      const pending = addMessage("助手", "正在同步 docs 目录中的 PDF/TXT 到知识库，Excel 会保留给表格分析...");
      try {
        const res = await fetch("/ingest-docs", { method: "POST" });
        const data = await res.json();
        pending.innerHTML = `<span class="role">助手 · ingest</span>${renderMarkdown(data.answer || "")}`;
        checkHealth();
      } catch (error) {
        pending.innerHTML = `<span class="role">助手</span>同步失败：${escapeHtml(error.message)}`;
      }
    });

    document.querySelector("#analysis-upload-button").addEventListener("click", () => {
      analysisFile.click();
    });

    analysisFile.addEventListener("change", async () => {
      if (!analysisFile.files.length) return;
      const formData = new FormData();
      formData.append("session_id", sessionId);
      Array.from(analysisFile.files).forEach((file) => formData.append("file", file));
      const uploadNames = Array.from(analysisFile.files).map((file) => file.name).join("、");
      const previousPlaceholder = question.placeholder;
      question.placeholder = `正在附加：${uploadNames}`;
      send.disabled = true;
      try {
        const res = await fetch("/upload-analysis-file", { method: "POST", body: formData });
        const data = await res.json();
        const files = data.files || (data.file ? [data.file] : []);
        for (const file of files) {
          if (file.file_name && !pendingAnalysisFiles.some((pendingFile) => pendingFile.file_name === file.file_name)) {
            pendingAnalysisFiles.push(file);
          }
        }
        renderComposerAttachments();
        await refreshSessionFiles();
        await checkHealth();
      } catch (error) {
        addMessage("助手", `临时分析文件上传失败：${escapeHtml(error.message)}`);
      } finally {
        send.disabled = false;
        question.placeholder = previousPlaceholder;
        analysisFile.value = "";
        question.focus();
      }
    });

    document.querySelector("#knowledge-upload").addEventListener("click", () => {
      knowledgeFile.click();
    });

    knowledgeFile.addEventListener("change", async () => {
      if (!knowledgeFile.files.length) return;
      const formData = new FormData();
      formData.append("file", knowledgeFile.files[0]);
      const pending = addMessage("助手", `正在上传资料到知识库：${knowledgeFile.files[0].name}`);
      try {
        const res = await fetch("/upload-knowledge-file", { method: "POST", body: formData });
        const data = await res.json();
        pending.innerHTML = `<span class="role">助手 · knowledge-file</span>${renderMarkdown(data.answer || "")}`;
        await checkHealth();
      } catch (error) {
        pending.innerHTML = `<span class="role">助手</span>上传失败：${escapeHtml(error.message)}`;
      } finally {
        knowledgeFile.value = "";
      }
    });

    document.querySelector("#clear-history").addEventListener("click", async () => {
      if (!confirm("确认清空当前会话历史、记忆和临时上传文件吗？知识库不会被删除。")) return;
      const pending = addMessage("助手", "正在清空当前会话历史...");
      try {
        const res = await fetch("/clear-history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId })
        });
        const data = await res.json();
        pending.innerHTML = `<span class="role">助手 · clear-history</span>${renderMarkdown(data.answer || "")}`;
        pendingAnalysisFiles = [];
        await refreshSessionFiles();
        await checkHealth();
      } catch (error) {
        pending.innerHTML = `<span class="role">助手</span>清空失败：${escapeHtml(error.message)}`;
      }
    });

    document.querySelector("#clear-knowledge").addEventListener("click", async () => {
      if (!confirm("确认清空知识库吗？这会删除后台知识库向量和 md5 入库记录，但不会删除 docs 原始文件。")) return;
      if (!confirm("二次确认：清空后需要重新上传或同步资料才能使用知识库问答。继续吗？")) return;
      const pending = addMessage("助手", "正在清空知识库向量...");
      try {
        const res = await fetch("/clear-knowledge-base", { method: "POST" });
        const data = await res.json();
        pending.innerHTML = `<span class="role">助手 · clear-knowledge</span>${renderMarkdown(data.answer || "")}`;
        await checkHealth();
      } catch (error) {
        pending.innerHTML = `<span class="role">助手</span>清空失败：${escapeHtml(error.message)}`;
      }
    });

    checkHealth();
    refreshSessionFiles();
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    """统一输出 JSON 响应，避免每个接口重复设置响应头。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, status: int, html: str) -> None:
    """输出首页 HTML。"""
    body = html.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _parse_multipart(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    """
    解析浏览器 FormData 上传。

    项目当前使用标准库 http.server，没有 FastAPI/Werkzeug，所以这里实现一个足够覆盖
    本地单文件上传场景的 multipart 解析器。
    """
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        raise ValueError("请求必须使用 multipart/form-data")

    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    raw_body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
    delimiter = ("--" + boundary).encode("utf-8")
    result: dict[str, Any] = {"fields": {}, "files": []}

    for part in raw_body.split(delimiter):
        part = part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        header_blob, _, content = part.partition(b"\r\n\r\n")
        if not header_blob:
            continue
        headers = header_blob.decode("utf-8", errors="ignore").split("\r\n")
        disposition = next((line for line in headers if line.lower().startswith("content-disposition:")), "")
        name = ""
        filename = ""
        for item in disposition.split(";"):
            item = item.strip()
            if item.startswith("name="):
                name = item.split("=", 1)[1].strip('"')
            elif item.startswith("filename="):
                filename = item.split("=", 1)[1].strip('"')
        if content.endswith(b"\r\n"):
            content = content[:-2]
        if filename:
            result["files"].append({"name": name, "filename": filename, "content": content})
        elif name:
            result["fields"][name] = content.decode("utf-8", errors="ignore")
    return result


def _check_mongodb() -> bool:
    """检查 MongoDB 是否可连接；失败时只返回 False，不影响报表功能。"""
    try:
        import pymongo

        client = pymongo.MongoClient(info.mongodb_url, serverSelectionTimeoutMS=800)
        client.admin.command("ping")
        client.close()
        return True
    except Exception:
        return False


def _check_vector_store() -> bool:
    """检查本地 Chroma 持久化目录是否存在，用于前端健康状态展示。"""
    return Path(info.persist_directory).exists()


def _looks_like_current_attachment_question(question: str) -> bool:
    """
    判断用户是不是在问“这个表/附件/刚上传文件”。

    如果用户没有随本条消息提交附件，这类问题不能再自动拿会话历史文件分析，
    否则就会出现用户明明想分析新附件，系统却先分析旧附件的串场问题。
    """
    normalized = question.lower()
    attachment_words = ["这个表", "这张表", "附件", "上传", "文件", "表格", "这个excel", "这个xlsx", "刚才传"]
    analysis_words = ["分析", "总结", "统计", "异常", "建议", "看看", "生成"]
    return any(word in normalized for word in attachment_words) and any(word in normalized for word in analysis_words)


class RagHttpHandler(BaseHTTPRequestHandler):
    """标准库 HTTP 请求处理器，提供首页、/chat 和 /health。"""

    server_version = "LocalRagMvp/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        """减少终端噪声，仅保留必要访问日志。"""
        print("[web]", format % args)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            _html_response(self, 200, HTML_PAGE)
            return
        if parsed.path == "/health":
            session_id = query.get("session_id", [info.default_session_id])[0]
            vector_service = VectorRetrieveService()
            session_service = SessionFileService()
            _json_response(
                self,
                200,
                {
                    "ok": True,
                    "mongodb": _check_mongodb(),
                    "vector_store": _check_vector_store(),
                    "docs_dir": str(info.docs_dir),
                    "knowledge_count": vector_service.knowledge_count(),
                    "history_count": vector_service.history_count(),
                    "session_file_count": len(session_service.list_files(session_id)),
                },
            )
            return
        if parsed.path == "/session-files":
            session_id = query.get("session_id", [info.default_session_id])[0]
            files = SessionFileService().summarize_session(session_id)
            _json_response(self, 200, {"ok": True, "files": files})
            return
        _json_response(self, 404, {"ok": False, "error": "接口不存在"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in {"/chat", "/chat-stream"}:
            if parsed.path == "/ingest-docs":
                self._handle_ingest_docs()
                return
            if parsed.path == "/upload-analysis-file":
                self._handle_upload_analysis_file()
                return
            if parsed.path == "/upload-knowledge-file":
                self._handle_upload_knowledge_file()
                return
            if parsed.path == "/clear-history":
                self._handle_clear_history()
                return
            if parsed.path == "/clear-knowledge-base":
                self._handle_clear_knowledge_base()
                return
            _json_response(self, 404, {"ok": False, "error": "接口不存在"})
            return

        if parsed.path == "/chat-stream":
            self._handle_chat_stream()
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question", "")).strip()
            session_id = str(payload.get("session_id", info.default_session_id)).strip() or info.default_session_id
            raw_analysis_files = payload.get("analysis_files", [])
            analysis_files = [safe_filename(str(name)) for name in raw_analysis_files if str(name).strip()] if isinstance(raw_analysis_files, list) else []
            if not question:
                _json_response(self, 400, {"ok": False, "error": "question不能为空"})
                return

            if not analysis_files and _looks_like_current_attachment_question(question):
                answer = (
                    "# 请先附加本次要分析的文件\n\n"
                    "你这句话像是在分析“当前上传的表格”，但本条消息没有带附件。\n\n"
                    "请点输入框左侧的 `+` 选择 Excel/CSV，然后在同一条消息里输入你的问题再发送。"
                )
            else:
                # 普通请求统一进入 LangGraph，由模型结构化路由节点判断 knowledge_qa/data_report/pure_chat。
                from rag import RagService

                answer = RagService().ask(
                    question=question,
                    session_id=session_id,
                    analysis_file_names=analysis_files,
                )

            _json_response(self, 200, {"ok": True, "answer": answer, "type": "assistant"})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "answer": f"服务处理失败：{exc}", "type": "error"})

    def _handle_chat_stream(self) -> None:
        """
        流式聊天接口。

        使用 NDJSON（一行一个 JSON 对象）而不是 SSE，是为了让标准库 http.server 下的
        POST 请求和前端 fetch ReadableStream 更简单稳定。每个 chunk 都会尽快 flush。
        """
        headers_sent = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            question = str(payload.get("question", "")).strip()
            session_id = str(payload.get("session_id", info.default_session_id)).strip() or info.default_session_id
            raw_analysis_files = payload.get("analysis_files", [])
            analysis_files = [safe_filename(str(name)) for name in raw_analysis_files if str(name).strip()] if isinstance(raw_analysis_files, list) else []
            if not question:
                _json_response(self, 400, {"ok": False, "answer": "question不能为空", "type": "error"})
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            headers_sent = True

            def write_event(payload_dict: dict[str, Any]) -> None:
                line = json.dumps(payload_dict, ensure_ascii=False) + "\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            if not analysis_files and _looks_like_current_attachment_question(question):
                write_event({
                    "type": "assistant",
                    "delta": "# 请先附加本次要分析的文件\n\n你这句话像是在分析“当前上传的表格”，但本条消息没有带附件。\n\n请点输入框左侧的 `+` 选择 Excel/CSV，然后在同一条消息里输入你的问题再发送。",
                })
                write_event({"type": "done"})
                return

            from rag import RagService

            wrote_any = False
            for chunk in RagService().stream_ask(
                question=question,
                session_id=session_id,
                analysis_file_names=analysis_files,
            ):
                if chunk:
                    wrote_any = True
                    write_event({"type": "assistant", "delta": chunk})
            if not wrote_any:
                write_event({"type": "assistant", "delta": ""})
            write_event({"type": "done"})
        except Exception as exc:
            if not headers_sent:
                _json_response(self, 500, {"ok": False, "answer": f"服务处理失败：{exc}", "type": "error"})
                return
            try:
                line = json.dumps({"type": "error", "delta": f"服务处理失败：{exc}"}, ensure_ascii=False) + "\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def _handle_ingest_docs(self) -> None:
        """处理网页端的知识库同步请求。"""
        try:
            from knowledge_base import KnowledgeBaseService

            results = KnowledgeBaseService().upload_docs_dir(info.docs_dir)
            answer = "# docs 知识库同步结果\n\n" + "\n".join(f"- {line}" for line in results)
            _json_response(self, 200, {"ok": True, "answer": answer, "type": "ingest"})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "answer": f"知识库同步失败：{exc}", "type": "ingest"})

    def _handle_upload_analysis_file(self) -> None:
        """处理聊天框 + 上传的临时分析文件。"""
        try:
            parsed = _parse_multipart(self)
            session_id = parsed["fields"].get("session_id", info.default_session_id)
            if not parsed["files"]:
                raise ValueError("没有收到上传文件")
            session_service = SessionFileService()
            summaries = [
                session_service.save_analysis_file(
                    session_id=session_id,
                    filename=upload["filename"],
                    content=upload["content"],
                )
                for upload in parsed["files"]
            ]

            lines = ["# 临时分析文件已附加到下一条消息", ""]
            for summary in summaries:
                lines.extend(
                    [
                        f"## {summary['file_name']}",
                        "",
                        f"- 数据行数：{summary['row_count']}",
                        f"- 字段数量：{summary['column_count']}",
                        f"- 字段：{', '.join(summary['columns']) or '未识别'}",
                        "",
                    ]
                )
            lines.append("现在在输入框里写下问题并发送，本次分析只会针对这些待发送附件；历史上传文件只作为参考列表保留。")
            _json_response(self, 200, {"ok": True, "answer": "\n".join(lines), "files": summaries, "type": "analysis_file"})
        except Exception as exc:
            _json_response(self, 400, {"ok": False, "answer": f"临时分析文件上传失败：{exc}", "type": "analysis_file"})

    def _handle_upload_knowledge_file(self) -> None:
        """处理资料上传：只允许 PDF/TXT 写入长期知识库。"""
        try:
            parsed = _parse_multipart(self)
            if not parsed["files"]:
                raise ValueError("没有收到上传文件")
            upload = parsed["files"][0]
            filename = safe_filename(upload["filename"])
            suffix = Path(filename).suffix.lower()
            if suffix in ANALYSIS_SUFFIXES:
                raise ValueError("Excel/CSV 请使用聊天框旁边的 + 做临时分析，不要上传到知识库")
            if suffix not in {".pdf", ".txt"}:
                raise ValueError("知识库第一版仅支持 .pdf 和 .txt")

            target = info.docs_dir / filename
            target.write_bytes(upload["content"])

            from knowledge_base import KnowledgeBaseService

            result = KnowledgeBaseService().upload_by_file(str(target))
            answer = f"# 资料上传结果\n\n- 文件名：{filename}\n- 入库结果：{result}"
            _json_response(self, 200, {"ok": True, "answer": answer, "type": "knowledge_file"})
        except Exception as exc:
            _json_response(self, 400, {"ok": False, "answer": f"资料上传失败：{exc}", "type": "knowledge_file"})

    def _handle_clear_history(self) -> None:
        """清空当前会话历史、远期记忆和临时上传文件。"""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            session_id = str(payload.get("session_id", info.default_session_id)).strip() or info.default_session_id
            deleted_files = SessionFileService().clear_session(session_id)

            try:
                from rag import RagService

                RagService().clear_history(session_id=session_id)
                history_result = "MongoDB 图状态和历史向量已清空"
            except Exception as exc:
                history_result = f"历史清理部分失败：{exc}"

            answer = (
                "# 当前会话历史清理完成\n\n"
                f"- 会话 ID：{session_id}\n"
                f"- 临时分析文件删除数量：{deleted_files}\n"
                f"- 历史状态：{history_result}\n"
                "- 知识库：未删除"
            )
            _json_response(self, 200, {"ok": True, "answer": answer, "type": "clear_history"})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "answer": f"清空当前会话历史失败：{exc}", "type": "clear_history"})

    def _handle_clear_knowledge_base(self) -> None:
        """清空知识库向量集合和 md5 入库记录。"""
        try:
            result = VectorRetrieveService().delete_knowledge_base()
            answer = (
                "# 知识库已清空\n\n"
                f"- 删除知识库向量：{result['deleted_vectors']} 条\n"
                f"- 删除 md5 入库记录文件：{result['deleted_md5_files']} 个\n"
                "- docs 原始文件：未删除\n"
                "- 当前会话历史：未删除"
            )
            _json_response(self, 200, {"ok": True, "answer": answer, "type": "clear_knowledge"})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "answer": f"清空知识库失败：{exc}", "type": "clear_knowledge"})


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动本地网页服务。"""
    server = ThreadingHTTPServer((host, port), RagHttpHandler)
    print(f"本地 RAG 小助手已启动：http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
