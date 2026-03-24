"""
rlm_visualizer.py — Self-contained HTML visualizer for RLM execution traces
============================================================================

Generates a single HTML file (with embedded JS/CSS) that lets you
interactively explore the recursive call tree produced by
:class:`RLMAgent.completion`.

Usage from a notebook or script::

    from rlm_smolagent import RLMAgent
    from rlm_visualizer import save_html, save_json, load_json

    agent = RLMAgent(...)
    result = agent.completion(task="...", context="...", capture_prompt_traces=True)

    # Save interactive HTML visualization
    save_html(result, "trace.html")

    # Save raw JSON for later analysis / re-visualization
    save_json(result, "trace.json")

    # Re-create HTML from a previously saved JSON
    data = load_json("trace.json")
    save_html(data, "trace_reloaded.html")

The HTML file requires **no server** — open it directly in any browser.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _completion_to_dict(result: Any) -> dict[str, Any]:
    """
    Accept either an ``RLMCompletion`` instance or a plain ``dict``
    (previously saved via :func:`save_json`) and return a JSON-serialisable
    dictionary with at least ``response`` and ``metadata`` keys.
    """
    if isinstance(result, dict):
        return result
    # Quacks like RLMCompletion
    return {
        "response": getattr(result, "response", ""),
        "metadata": getattr(result, "metadata", {}),
    }


def save_json(result: Any, path: str | Path) -> Path:
    """Persist the full RLMCompletion payload as JSON."""
    path = Path(path)
    data = _completion_to_dict(result)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a previously saved RLMCompletion JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _count_nodes(node: dict) -> int:
    """Count total nodes in the call tree."""
    return 1 + sum(_count_nodes(c) for c in node.get("children", []))


def _max_depth(node: dict) -> int:
    """Find maximum depth in the call tree."""
    children = node.get("children", [])
    if not children:
        return node.get("depth", 0)
    return max(_max_depth(c) for c in children)


def _total_llm_requests(node: dict) -> int:
    """Count total LLM requests across the tree."""
    count = len(node.get("llm_requests", []))
    for child in node.get("children", []):
        count += _total_llm_requests(child)
    return count


def _total_duration(node: dict) -> float:
    """Get the root node's duration."""
    return node.get("duration_s", 0)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RLM Trace Visualizer</title>
<style>
/* ---- Reset & base ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0f1117;
  --bg-card: #1a1d27;
  --bg-card-hover: #22263a;
  --border: #2a2e3e;
  --text: #e4e4e7;
  --text-muted: #71717a;
  --primary: #6366f1;
  --primary-light: #818cf8;
  --green: #22c55e;
  --yellow: #eab308;
  --cyan: #06b6d4;
  --magenta: #d946ef;
  --red: #ef4444;
  --orange: #f97316;
  --code-bg: #0d0f14;
  --role-system: #64748b;
  --role-user: #3b82f6;
  --role-assistant: #22c55e;
  --role-tool: #f97316;
  --font-mono: 'SF Mono', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', Consolas, monospace;
  --font-sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.light {
  --bg: #fafafa;
  --bg-card: #ffffff;
  --bg-card-hover: #f4f4f5;
  --border: #e4e4e7;
  --text: #18181b;
  --text-muted: #71717a;
  --code-bg: #f4f4f5;
}

body {
  font-family: var(--font-sans);
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  overflow: hidden;
  height: 100vh;
}

/* ---- Layout ---- */
.app { display: flex; flex-direction: column; height: 100vh; }

.header {
  padding: 12px 24px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
  background: var(--bg-card);
}
.header h1 { font-size: 18px; font-weight: 700; }
.header h1 .accent { color: var(--primary); }
.header h1 .sub { color: var(--text-muted); font-weight: 400; margin-left: 8px; }
.header-right { display: flex; align-items: center; gap: 12px; }

.stats-bar {
  display: flex;
  gap: 8px;
  padding: 10px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  flex-shrink: 0;
  flex-wrap: wrap;
}

.stat-card {
  padding: 8px 16px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg);
  min-width: 100px;
  text-align: center;
}
.stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }
.stat-value { font-size: 20px; font-weight: 700; font-family: var(--font-mono); }
.stat-value.cyan { color: var(--cyan); }
.stat-value.green { color: var(--green); }
.stat-value.magenta { color: var(--magenta); }
.stat-value.yellow { color: var(--yellow); }

.answer-bar {
  padding: 10px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
  flex-shrink: 0;
}
.answer-bar .label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 4px; }
.answer-bar .value { font-size: 14px; color: var(--green); font-weight: 500; }

.main-panels {
  flex: 1;
  display: flex;
  min-height: 0;
  overflow: hidden;
}

/* ---- Tree Panel (left) ---- */
.tree-panel {
  width: 400px;
  min-width: 280px;
  max-width: 50vw;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  background: var(--bg-card);
  flex-shrink: 0;
  resize: horizontal;
  overflow: auto;
}
.tree-panel-header {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: 6px;
}
.tree-content {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

/* ---- Tree node ---- */
.tree-node { margin-left: 0; }
.tree-node .children { margin-left: 20px; border-left: 1px solid var(--border); padding-left: 4px; }

.node-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  transition: background 0.15s;
  border: 1px solid transparent;
}
.node-row:hover { background: var(--bg-card-hover); }
.node-row.selected {
  background: var(--bg-card-hover);
  border-color: var(--primary);
}

.node-toggle {
  width: 18px;
  height: 18px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 10px;
  color: var(--text-muted);
  border: none;
  background: none;
  cursor: pointer;
  flex-shrink: 0;
  border-radius: 4px;
  transition: background 0.15s, color 0.15s;
}
.node-toggle:hover { background: var(--border); color: var(--text); }

.node-depth {
  font-family: var(--font-mono);
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--primary);
  color: #fff;
  flex-shrink: 0;
}

.node-task {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 12px;
}

.node-duration {
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
  flex-shrink: 0;
}

.node-badge {
  font-size: 9px;
  padding: 1px 5px;
  border-radius: 3px;
  flex-shrink: 0;
  font-weight: 600;
}
.node-badge.leaf { background: var(--green); color: #fff; }
.node-badge.recursive { background: var(--magenta); color: #fff; }

/* ---- Detail Panel (right) ---- */
.detail-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  overflow: hidden;
}
.detail-panel-header {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
  background: var(--bg-card);
}
.detail-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
.detail-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-muted);
  font-size: 14px;
}

.detail-section {
  margin-bottom: 20px;
}
.detail-section-title {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  user-select: none;
}
.detail-section-title .chevron {
  font-size: 10px;
  transition: transform 0.15s;
}
.detail-section-title .chevron.collapsed { transform: rotate(-90deg); }
.detail-section-body { }
.detail-section-body.collapsed { display: none; }

.detail-field {
  display: grid;
  grid-template-columns: 100px 1fr;
  gap: 4px;
  margin-bottom: 6px;
  font-size: 13px;
}
.detail-field .label { color: var(--text-muted); font-size: 11px; }
.detail-field .value { word-break: break-word; }

.detail-text-block {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 400px;
  overflow-y: auto;
}

/* ---- LLM Messages ---- */
.llm-request {
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 12px;
  overflow: hidden;
}
.llm-request-header {
  padding: 8px 12px;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
  font-size: 11px;
  font-family: var(--font-mono);
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}
.llm-request-header:hover { background: var(--bg-card-hover); }

.phase-badge {
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 3px;
  font-weight: 600;
}
.phase-badge.agent_step { background: var(--primary); color: #fff; }
.phase-badge.plain_completion { background: var(--orange); color: #fff; }

.llm-messages { padding: 8px; }

.message-bubble {
  margin-bottom: 8px;
  border-radius: 8px;
  border: 1px solid var(--border);
  overflow: hidden;
}
.message-role {
  padding: 4px 10px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.message-role.system { background: var(--role-system); color: #fff; }
.message-role.user { background: var(--role-user); color: #fff; }
.message-role.assistant { background: var(--role-assistant); color: #fff; }
.message-role.tool-call { background: var(--role-tool); color: #fff; }
.message-role.tool-response { background: var(--role-tool); color: #fff; }

.message-content {
  padding: 10px;
  font-size: 12px;
  font-family: var(--font-mono);
  white-space: pre-wrap;
  word-break: break-word;
  max-height: 300px;
  overflow-y: auto;
  line-height: 1.6;
  background: var(--code-bg);
}

/* ---- Theme toggle ---- */
.theme-btn {
  background: var(--bg);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
  transition: background 0.15s;
}
.theme-btn:hover { background: var(--bg-card-hover); }

/* ---- Scrollbars ---- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ---- Keyboard hint footer ---- */
.footer {
  padding: 4px 24px;
  border-top: 1px solid var(--border);
  font-size: 10px;
  color: var(--text-muted);
  text-align: center;
  flex-shrink: 0;
  background: var(--bg-card);
}
kbd {
  padding: 1px 5px;
  border-radius: 3px;
  background: var(--bg);
  border: 1px solid var(--border);
  font-family: var(--font-mono);
  font-size: 9px;
}

/* ---- Empty state ---- */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--text-muted);
  gap: 8px;
}
.empty-state .icon { font-size: 32px; opacity: 0.5; }
</style>
</head>
<body>
<div class="app" id="app">
  <header class="header">
    <h1><span class="accent">RLM</span><span class="sub">Trace Visualizer</span></h1>
    <div class="header-right">
      <span style="font-family:var(--font-mono);font-size:11px;color:var(--text-muted);" id="gen-time"></span>
      <button class="theme-btn" onclick="toggleTheme()" title="Toggle light/dark theme">🌓 Theme</button>
    </div>
  </header>

  <div class="stats-bar" id="stats-bar"></div>

  <div class="answer-bar" id="answer-bar">
    <div class="label">Final Answer</div>
    <div class="value" id="final-answer"></div>
  </div>

  <div class="main-panels">
    <div class="tree-panel">
      <div class="tree-panel-header">◈ Recursive Call Tree</div>
      <div class="tree-content" id="tree-content"></div>
    </div>
    <div class="detail-panel">
      <div class="detail-panel-header">▤ Node Details</div>
      <div class="detail-content" id="detail-content">
        <div class="detail-placeholder">← Select a node from the tree to inspect</div>
      </div>
    </div>
  </div>

  <div class="footer">
    <kbd>↑</kbd> <kbd>↓</kbd> Navigate tree &nbsp;·&nbsp;
    <kbd>←</kbd> <kbd>→</kbd> Collapse / Expand &nbsp;·&nbsp;
    <kbd>Enter</kbd> Select node
  </div>
</div>

<script>
// ---- Embedded trace data ----
const TRACE_DATA = __TRACE_JSON__;

const callTree = (TRACE_DATA.metadata || {}).call_tree || {};
const response = TRACE_DATA.response || "";

// ---- Flatten nodes for keyboard navigation ----
let allNodes = [];
let selectedNodeId = null;
let collapsedNodes = new Set();

function flattenTree(node, path) {
  const id = path;
  allNodes.push({ id, node, path });
  (node.children || []).forEach((child, i) => {
    flattenTree(child, path + "." + i);
  });
}

function rebuildFlatList() {
  allNodes = [];
  flattenTree(callTree, "0");
}
rebuildFlatList();

// ---- Visible node list (respecting collapsed state) ----
function getVisibleNodes() {
  const visible = [];
  function walk(node, path) {
    visible.push({ id: path, node, path });
    if (!collapsedNodes.has(path)) {
      (node.children || []).forEach((child, i) => {
        walk(child, path + "." + i);
      });
    }
  }
  walk(callTree, "0");
  return visible;
}

// ---- Stats ----
function countNodes(n) { return 1 + (n.children || []).reduce((s, c) => s + countNodes(c), 0); }
function maxDepth(n) { return (n.children || []).length === 0 ? (n.depth || 0) : Math.max(...(n.children || []).map(maxDepth)); }
function countRequests(n) { return (n.llm_requests || []).length + (n.children || []).reduce((s, c) => s + countRequests(c), 0); }

function renderStats() {
  const bar = document.getElementById("stats-bar");
  const stats = [
    { label: "Total Nodes", value: countNodes(callTree), cls: "cyan" },
    { label: "Max Depth", value: maxDepth(callTree), cls: "magenta" },
    { label: "LLM Requests", value: countRequests(callTree), cls: "green" },
    { label: "Duration", value: (callTree.duration_s || 0).toFixed(2) + "s", cls: "yellow" },
    { label: "Root Context", value: formatBytes(callTree.context_size || 0), cls: "cyan" },
  ];
  bar.innerHTML = stats.map(s => `
    <div class="stat-card">
      <div class="stat-label">${s.label}</div>
      <div class="stat-value ${s.cls}">${s.value}</div>
    </div>
  `).join("");
}

function formatBytes(bytes) {
  if (bytes === 0) return "0";
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + " KB";
  return (bytes/1024/1024).toFixed(1) + " MB";
}

// ---- Tree rendering ----
function esc(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderTree() {
  document.getElementById("tree-content").innerHTML = renderTreeNode(callTree, "0");
}

function renderTreeNode(node, path) {
  const hasChildren = (node.children || []).length > 0;
  const isCollapsed = collapsedNodes.has(path);
  const isSelected = selectedNodeId === path;
  const isLeaf = !hasChildren;
  const duration = (node.duration_s || 0).toFixed(2);

  let html = '<div class="tree-node">';
  html += `<div class="node-row ${isSelected ? "selected" : ""}" data-path="${esc(path)}" onclick="selectNode('${esc(path)}')">`;

  if (hasChildren) {
    html += `<button class="node-toggle" onclick="event.stopPropagation();toggleCollapse('${esc(path)}')">${isCollapsed ? "▶" : "▼"}</button>`;
  } else {
    html += '<span style="width:18px;display:inline-block"></span>';
  }

  html += `<span class="node-depth">D${node.depth || 0}</span>`;
  html += `<span class="node-task" title="${esc(node.task_preview || "")}">${esc(node.task_preview || "(no task)")}</span>`;

  if (isLeaf) {
    html += '<span class="node-badge leaf">leaf</span>';
  } else {
    html += `<span class="node-badge recursive">${(node.children || []).length} child${(node.children||[]).length>1?"ren":""}</span>`;
  }
  html += `<span class="node-duration">${duration}s</span>`;
  html += '</div>';

  if (hasChildren && !isCollapsed) {
    html += '<div class="children">';
    (node.children || []).forEach((child, i) => {
      html += renderTreeNode(child, path + "." + i);
    });
    html += '</div>';
  }

  html += '</div>';
  return html;
}

function toggleCollapse(path) {
  if (collapsedNodes.has(path)) {
    collapsedNodes.delete(path);
  } else {
    collapsedNodes.add(path);
  }
  renderTree();
}

function selectNode(path) {
  selectedNodeId = path;
  renderTree();
  renderDetail(path);
}

// ---- Navigate to node by path ----
function getNodeByPath(path) {
  const parts = path.split(".");
  let node = callTree;
  for (let i = 1; i < parts.length; i++) {
    node = (node.children || [])[parseInt(parts[i])];
    if (!node) return null;
  }
  return node;
}

// ---- Detail rendering ----
function renderDetail(path) {
  const node = getNodeByPath(path);
  if (!node) return;

  const panel = document.getElementById("detail-content");
  let h = "";

  // Task section
  h += renderDetailSection("Task", `<div class="detail-text-block">${esc(node.task_preview || "")}</div>`, true);

  // Metadata fields
  h += '<div class="detail-section">';
  h += `<div class="detail-field"><span class="label">Depth</span><span class="value">${node.depth || 0}</span></div>`;
  h += `<div class="detail-field"><span class="label">Duration</span><span class="value">${(node.duration_s || 0).toFixed(3)}s</span></div>`;
  h += `<div class="detail-field"><span class="label">Context Size</span><span class="value">${formatBytes(node.context_size || 0)}</span></div>`;
  h += `<div class="detail-field"><span class="label">Children</span><span class="value">${(node.children || []).length}</span></div>`;
  h += `<div class="detail-field"><span class="label">LLM Requests</span><span class="value">${(node.llm_requests || []).length}</span></div>`;
  h += '</div>';

  // Response section
  h += renderDetailSection("Response", `<div class="detail-text-block">${esc(node.response_preview || "")}</div>`, true);

  // LLM requests
  const requests = node.llm_requests || [];
  if (requests.length > 0) {
    let reqHtml = "";
    requests.forEach((req, i) => {
      reqHtml += renderLLMRequest(req, i);
    });
    h += renderDetailSection(`LLM Requests (${requests.length})`, reqHtml, true);
  }

  panel.innerHTML = h;
}

function renderDetailSection(title, bodyHtml, startOpen) {
  const id = "sec-" + title.replace(/\W+/g, "-").toLowerCase();
  return `<div class="detail-section">
    <div class="detail-section-title" onclick="toggleSection('${id}')">
      <span class="chevron ${startOpen ? '' : 'collapsed'}" id="${id}-chev">▼</span>
      ${esc(title)}
    </div>
    <div class="detail-section-body ${startOpen ? '' : 'collapsed'}" id="${id}-body">
      ${bodyHtml}
    </div>
  </div>`;
}

function toggleSection(id) {
  const chev = document.getElementById(id + "-chev");
  const body = document.getElementById(id + "-body");
  if (!chev || !body) return;
  chev.classList.toggle("collapsed");
  body.classList.toggle("collapsed");
}

function renderLLMRequest(req, index) {
  const phase = req.phase || "unknown";
  const depth = req.depth != null ? req.depth : "?";
  const step = req.node_step != null ? req.node_step : "?";
  const tools = (req.tool_names || []).join(", ") || "none";
  const id = "llm-req-" + index;
  const messages = req.messages || [];

  let h = `<div class="llm-request">`;
  h += `<div class="llm-request-header" onclick="toggleSection('${id}')">`;
  h += `<span class="chevron" id="${id}-chev">▼</span>`;
  h += `<span class="phase-badge ${esc(phase)}">${esc(phase)}</span>`;
  h += `<span>step ${step} · depth ${depth} · tools: ${esc(tools)}</span>`;
  h += `<span style="margin-left:auto;font-size:9px;color:var(--text-muted)">${messages.length} msg${messages.length!==1?"s":""}</span>`;
  h += `</div>`;
  h += `<div class="llm-messages" id="${id}-body">`;

  messages.forEach(msg => {
    h += renderMessage(msg);
  });

  h += `</div></div>`;
  return h;
}

function renderMessage(msg) {
  const role = (msg.role || "unknown").toLowerCase();
  let displayRole = role;
  const content = msg.content || "";

  // Handle tool calls and tool responses from smolagents
  if (role === "assistant" && msg.tool_calls) {
    displayRole = "tool-call";
  } else if (role === "tool") {
    displayRole = "tool-response";
  }

  let h = `<div class="message-bubble">`;
  h += `<div class="message-role ${esc(displayRole)}">${esc(displayRole)}</div>`;
  h += `<div class="message-content">${esc(typeof content === "string" ? content : JSON.stringify(content, null, 2))}</div>`;
  h += `</div>`;
  return h;
}

// ---- Theme toggle ----
let isDark = true;
function toggleTheme() {
  isDark = !isDark;
  document.body.classList.toggle("light", !isDark);
}

// ---- Keyboard navigation ----
document.addEventListener("keydown", function(e) {
  const visible = getVisibleNodes();
  const idx = visible.findIndex(n => n.id === selectedNodeId);

  if (e.key === "ArrowDown" || e.key === "j") {
    e.preventDefault();
    const next = Math.min(idx + 1, visible.length - 1);
    if (next >= 0) selectNode(visible[next].id);
  } else if (e.key === "ArrowUp" || e.key === "k") {
    e.preventDefault();
    const prev = Math.max(idx - 1, 0);
    if (prev >= 0) selectNode(visible[prev].id);
  } else if (e.key === "ArrowLeft" || e.key === "h") {
    if (selectedNodeId && !collapsedNodes.has(selectedNodeId)) {
      const node = getNodeByPath(selectedNodeId);
      if (node && (node.children || []).length > 0) {
        e.preventDefault();
        collapsedNodes.add(selectedNodeId);
        renderTree();
      }
    }
  } else if (e.key === "ArrowRight" || e.key === "l") {
    if (selectedNodeId && collapsedNodes.has(selectedNodeId)) {
      e.preventDefault();
      collapsedNodes.delete(selectedNodeId);
      renderTree();
    }
  } else if (e.key === "Enter") {
    if (selectedNodeId) {
      e.preventDefault();
      renderDetail(selectedNodeId);
    }
  }
});

// ---- Init ----
renderStats();
document.getElementById("final-answer").textContent = response || "(no final answer)";
document.getElementById("gen-time").textContent = "__GENERATED_AT__";
renderTree();

// Auto-select root
if (allNodes.length > 0) {
  selectNode(allNodes[0].id);
}
</script>
</body>
</html>"""


def save_html(result: Any, path: str | Path) -> Path:
    """
    Generate a self-contained HTML visualizer from an RLMCompletion or dict.

    The resulting file can be opened in any browser without a server.

    Parameters
    ----------
    result:
        An ``RLMCompletion`` instance or a previously saved ``dict``
        (from :func:`save_json` / :func:`load_json`).
    path:
        Destination file path for the HTML.

    Returns
    -------
    Path
        The resolved path of the generated file.
    """
    path = Path(path)
    data = _completion_to_dict(result)

    # Serialise trace data to a JSON literal safe for embedding in <script>.
    # We escape </script> sequences to prevent premature tag closure.
    trace_json = json.dumps(data, default=str)
    trace_json = trace_json.replace("</", r"<\/")

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    html_str = _HTML_TEMPLATE.replace("__TRACE_JSON__", trace_json)
    html_str = html_str.replace("__GENERATED_AT__", html.escape(generated_at))

    path.write_text(html_str, encoding="utf-8")
    return path
