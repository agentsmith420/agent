#!/usr/bin/env python3
"""
Kimi — Full-Stack Coding Agent (bulletproof build)
- Formatted markdown output (rich)
- Visible reasoning + tool-step trace
- Image input: quoted, \\ -escaped, or bare paths with spaces all work
- Self-healing history: tool_call/tool_result pairs are ALWAYS valid
- ESC cancels ANYWHERE in a turn (global watcher thread — connecting,
  streaming, running tools, waiting on anything). Arrow keys ignored.
- Background tasks: long commands (installs, builds, training) run detached
  with logs + pid files; agent polls with task_status. Heavy commands are
  auto-converted so they can never hit the shell timeout.
- Todo lists: todo_write tool keeps a live checklist, shown in the terminal
  and replayed into the system prompt every turn.
- Memory: per-turn ledger (prompt/tool/answer steps), ISOLATED per agent
  process — a fresh agent starts with fresh memory. Replayed each turn as a
  compact MEMORY section so the model never re-does work.
- Slash commands: typing "/" shows a live-filtered menu below the prompt
  (arrow keys + tab to complete, like the web app's picker).
- Multi-model: /model picker · auto-failover on BalanceError
- Session log: everything persisted to ~/.kimi/ — crash → restart resumes
  full context automatically (machine JSONL + human-readable log)
"""
import os
import re, sys, json, base64, mimetypes, subprocess, difflib, shlex, time, queue, signal
import hashlib, atexit
from pathlib import Path
import threading, select
try:
    import termios, tty
except ImportError:
    termios = tty = None
try:
    import readline          # arrow-key history at the prompt (libedit on macOS)
except ImportError:
    readline = None

from openai import OpenAI
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text
from rich import box

console = Console()

# ─── Config ──────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("KIMI_API_KEY", "sk-ebbfd953e6554a86b21cfe07147a3efa")
BASE_URL = "https://ws-9el7am7u1tbjibzh.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"

# ─── Model registry (IDs confirmed from your /models endpoint) ───────────────
MODELS = {
    "k3":       "kimi-k3",
    "ds-pro":   "deepseek-v4-pro-0813",
    "glm":      "ZHIPU/GLM-5.3",
    "qwen27b":  "qwen3.8-27b",
    "qwen2.4t": "qwen3.8-2.4t-a95b",
    # non-chat models on this gateway (excluded from failover):
    #   qwen3.7-text-embedding · qwen-image-3.0-pro · qwen-image-3.0
}
FAILOVER_CHAIN = ["kimi-k3", "deepseek-v4-pro-0813", "ZHIPU/GLM-5.3",
                  "qwen3.8-27b", "qwen3.8-2.4t-a95b"]

def resolve_model(name: str) -> str:
    return MODELS.get(name, name)

MODEL         = resolve_model(os.environ.get("KIMI_MODEL", "k3"))
VISION_MODEL  = resolve_model(os.environ["KIMI_VISION_MODEL"]) if os.environ.get("KIMI_VISION_MODEL") else ""
MAX_TOOL_ROUNDS = 35
SHELL_TIMEOUT   = 120
CONFIRM_SHELL   = False
EXTRA_BODY      = {}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".heif"}

# Background tasks: detached processes with log/pid/label files, polled by taskId
BG_TASK_DIR = Path.home() / ".kimi" / "bg-tasks"
# Heavy commands auto-converted to background tasks (never hit shell timeout)
HEAVY_CMD_RE = re.compile(
    r"\b(pip3?|python3?\s+-m\s+pip)\s+install\b|\bnpm\s+(install|i)\b"
    r"|\byarn\s+(install|add)\b|\bpnpm\s+(install|add)\b"
    r"|\bapt(-get)?\s+install\b|\bgit\s+clone\b|\bwget\s|\bcurl\s+[^|;&]*-o\b"
    r"|\bmake\b|\bninja\b|\btrain\b|\bpython3?\s+[\w./-]+\.py\b")

DEAD_MODELS: set = set()     # session memory: BalanceError / unavailable models
NO_VISION:   set = set()     # models that rejected image input

client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=180.0, max_retries=1)
WORKDIR = Path.cwd()

# ─── Session logging (crash-safe resume) ─────────────────────────────────────
LOG_DIR     = Path.home() / ".kimi"
SESSION_DIR = LOG_DIR / "sessions"
HIST_FILE   = LOG_DIR / "history"          # readline input history

def _session_paths():
    sid = hashlib.md5(str(WORKDIR).encode()).hexdigest()[:12]
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR / f"{sid}.jsonl", LOG_DIR / f"{sid}.log"

def _jsonable_msg(m: dict) -> dict:
    """Make a message JSON-safe for the log: strip base64 image payloads."""
    out = {k: v for k, v in m.items()
           if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
    c = out.get("content")
    if isinstance(c, list):
        out["content"] = [
            {"type": "text", "text": "[image attached — payload omitted from log]"}
            if isinstance(p, dict) and p.get("type") == "image_url" else p
            for p in c]
    return out

def _human_line(kind, data) -> str:
    ts = time.strftime("%H:%M:%S")
    if kind == "msg":
        role = (data.get("role") or "?").upper()
        name = f" ({data.get('name')})" if data.get("name") else ""
        c = data.get("content")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        preview = re.sub(r"\s+", " ", str(c or ""))[:400]
        tcs = data.get("tool_calls") or []
        tc_txt = ""
        if tcs:
            tc_txt = " ⟶ " + "; ".join(
                f"{t.get('function',{}).get('name','?')}({str(t.get('function',{}).get('arguments',''))[:120]})"
                for t in tcs)
        return f"[{ts}] {role}{name}: {preview}{tc_txt}\n"
    if kind == "model":
        return f"[{ts}] MODEL → {data}\n"
    if kind in ("rewind", "trim", "reset"):
        return f"[{ts}] -- history {kind} ({data}) --\n"
    return f"[{ts}] {kind}: {str(data)[:300]}\n"

def log_event(kind: str, data):
    """Append to machine JSONL + human log. Never raises."""
    try:
        sf, hf = _session_paths()
        if kind == "msg":
            data = _jsonable_msg(data)
        with open(sf, "a") as f:
            f.write(json.dumps({"kind": kind, "data": data}, ensure_ascii=False) + "\n")
        with open(hf, "a") as f:
            f.write(_human_line(kind, data))
    except Exception:
        pass

def log_msg(m: dict):
    log_event("msg", m)

def load_session() -> list:
    """Replay the JSONL log → message list (excluding system). Exact: honors
    rewind/trim/reset events so cancelled turns stay cancelled."""
    sf, _ = _session_paths()
    if not sf.is_file():
        return []
    msgs = []
    try:
        lines = sf.read_text(errors="replace").splitlines()
    except Exception:
        return []
    for ln in lines:
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        k, d = rec.get("kind"), rec.get("data")
        if k == "msg" and isinstance(d, dict) and d.get("role") in ("user", "assistant", "tool"):
            msgs.append(d)
        elif k == "rewind":
            msgs = msgs[:max(0, int(d or 0))]
        elif k == "trim":
            msgs = msgs[-int(d or 50):]
        elif k == "reset":
            msgs = []
    return msgs

# ─── Memory ledger (per-agent, isolated per process) ──────────────────────────
# Mirrors api/src/services/agent/SessionMemory.ts: every turn records the user
# prompt, each tool call (args + result summary), and the final answer. On the
# next turn the ledger is replayed as a compact MEMORY section in the system
# prompt so the model always knows what it already did.
# ISOLATION: every agent process gets its OWN ledger file. A freshly spun-up
# agent starts with EMPTY memory — side-by-side agents never see each other's
# steps, so parallel sessions can't get confused. (KIMI_AGENT_ID overrides
# the generated id, e.g. for attaching tools to a known agent.)
AGENT_ID = re.sub(r"[^\w.-]", "_", os.environ.get("KIMI_AGENT_ID", "")) or \
           f"agent-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid():x}"
MEMORY_DIR  = LOG_DIR / "memory"
MEMORY_FILE = MEMORY_DIR / f"{AGENT_ID}.jsonl"

def _sweep_memory(max_age_days: int = 7):
    """Housekeeping: delete ledgers of agents that exited long ago."""
    try:
        cutoff = time.time() - max_age_days * 86400
        for p in MEMORY_DIR.glob("*.jsonl"):
            if p.stat().st_mtime < cutoff:
                p.unlink()
    except Exception:
        pass

_sweep_memory()   # housekeeping on startup: drop ledgers older than 7 days

def mem_record(kind: str, title: str, detail: str = "", result: str = "", ok: bool = True):
    """Append one step to the memory ledger. Never raises."""
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "kind": kind, "title": str(title)[:500],
               "detail": str(detail)[:2000], "result": str(result)[:2000], "ok": ok}
        with open(MEMORY_FILE, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

def mem_ledger(max_turns=6, full_turns=3):
    """Compact replay of the ledger for the system prompt. Recent turns keep
    full detail; older turns collapse to one line each."""
    try:
        if not MEMORY_FILE.is_file():
            return None
        lines = [json.loads(l) for l in MEMORY_FILE.read_text(errors="replace").splitlines()
                 if l.strip()]
        if not lines:
            return None
        # Group into turns: a 'prompt' starts a new turn
        turns, cur = [], []
        for r in lines:
            if r.get("kind") == "prompt":
                if cur: turns.append(cur)
                cur = [r]
            elif cur:
                cur.append(r)
        if cur: turns.append(cur)
        if not turns:
            return None
        turns = turns[-max_turns:]
        detailed_from = len(turns) - full_turns
        out = []
        for ti, turn in enumerate(turns):
            out.append(f"— Turn {ti+1} —")
            for r in turn:
                k = r.get("kind")
                if k == "prompt":
                    out.append(f"User: {r.get('title','')[:200]}")
                elif k == "tool":
                    status = "ok" if r.get("ok", True) else "FAILED"
                    if ti >= detailed_from:
                        args = f" ({r.get('detail','')[:120]})" if r.get("detail") else ""
                        res = f" → {r.get('result','')[:200]}" if r.get("result") else ""
                        out.append(f"Tool {r.get('title','')}{args} [{status}]{res}")
                    else:
                        out.append(f"Tool {r.get('title','')} [{status}]")
                elif k == "answer":
                    if ti >= detailed_from:
                        out.append(f"Answer: {r.get('result','')[:400]}")
                    else:
                        out.append("Answer: (given)")
        return "\n".join(out) if out else None
    except Exception:
        return None

def mem_clear():
    try:
        if MEMORY_FILE.is_file():
            MEMORY_FILE.unlink()
    except Exception:
        pass

# ─── Todos (live task list, Autohand-style) ───────────────────────────────────
# Mirrors api/src/services/agent/SessionTodos.ts: the model sends the COMPLETE
# updated list on every todo_write call. Rendered in the terminal and replayed
# into the system prompt each turn.
TODOS: list = []          # [{content, status, activeForm?}]

def normalize_todos(raw) -> list:
    if not isinstance(raw, list):
        return []
    out = []
    for t in raw[:30]:
        if not isinstance(t, dict):
            continue
        content = str(t.get("content") or t.get("title") or "")[:500]
        if not content:
            continue
        status = t.get("status")
        if status not in ("pending", "in_progress", "completed"):
            status = "pending"
        item = {"content": content, "status": status}
        if t.get("activeForm"):
            item["activeForm"] = str(t["activeForm"])[:200]
        out.append(item)
    return out

def render_todos() -> str:
    if not TODOS:
        return "(no active todos)"
    rows = []
    for i, t in enumerate(TODOS):
        mark = "[x]" if t["status"] == "completed" else "[~]" if t["status"] == "in_progress" else "[ ]"
        cur = "  ← CURRENT" if t["status"] == "in_progress" else ""
        rows.append(f"{i+1}. {mark} {t['content']}{cur}")
    return "\n".join(rows)

def print_todos_panel():
    if not TODOS:
        return
    rows = []
    for i, t in enumerate(TODOS):
        icon = "✓" if t["status"] == "completed" else "◐" if t["status"] == "in_progress" else "○"
        style = "green" if t["status"] == "completed" else "yellow" if t["status"] == "in_progress" else "dim"
        rows.append(f"[{style}]{icon} {t['content']}[/{style}]"
                    + (f" [dim]({t['activeForm']})[/]" if t.get("activeForm") and t["status"] == "in_progress" else ""))
    console.print(Panel("\n".join(rows), title="[bold]todos", box=box.ROUNDED,
                        border_style="cyan"))

# ─── Background tasks (detached, pollable) ─────────────────────────────────────
# Mirrors api/src/services/agent/tools.ts startBgTask: nohup + setsid so the
# process survives this agent; log/pid/label/exit files under ~/.kimi/bg-tasks.
def bg_start(command: str, label: str = "") -> dict:
    BG_TASK_DIR.mkdir(parents=True, exist_ok=True)
    task_id = f"task-{int(time.time()*1000):x}-{os.getpid():x}"
    task_label = label or command[:60]
    log_f = BG_TASK_DIR / f"{task_id}.log"
    pid_f = BG_TASK_DIR / f"{task_id}.pid"
    label_f = BG_TASK_DIR / f"{task_id}.label"
    exit_f = BG_TASK_DIR / f"{task_id}.exit"
    # Detached: setsid + nohup so it survives the agent process; write pid,
    # label, and exit code files next to the log for status checks.
    script = (f'echo $$ > {shlex.quote(str(pid_f))}; '
              f'echo {shlex.quote(task_label)} > {shlex.quote(str(label_f))}; '
              f'{command}; echo $? > {shlex.quote(str(exit_f))}')
    with open(log_f, "w") as lf:
        subprocess.Popen(["bash", "-c", script], cwd=WORKDIR,
                         stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return {"taskId": task_id, "label": task_label, "log": str(log_f)}

def bg_status(task_id: str, lines: int = 15) -> str:
    tid = re.sub(r"[^a-zA-Z0-9_-]", "", task_id)
    pid_f, exit_f = BG_TASK_DIR / f"{tid}.pid", BG_TASK_DIR / f"{tid}.exit"
    log_f, label_f = BG_TASK_DIR / f"{tid}.log", BG_TASK_DIR / f"{tid}.label"
    if not pid_f.is_file():
        return "status:unknown\nerror: no such task"
    out = []
    try:
        pid = int(pid_f.read_text().strip())
    except Exception:
        return "status:unknown\nerror: bad pid file"
    if exit_f.is_file():
        out.append(f"status:done exit_code:{exit_f.read_text().strip()}")
    else:
        try:
            os.kill(pid, 0)
            out.append(f"status:running pid:{pid}")
        except OSError:
            out.append(f"status:failed pid:{pid}")
    if label_f.is_file():
        out.append(f"label:{label_f.read_text().strip()}")
    out.append("--- log tail ---")
    try:
        tail = log_f.read_text(errors="replace").splitlines()[-lines:]
        out.extend(tail if tail else ["(no log yet)"])
    except Exception:
        out.append("(no log yet)")
    return "\n".join(out)

def bg_list() -> str:
    BG_TASK_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for pid_f in sorted(BG_TASK_DIR.glob("*.pid"), key=lambda p: p.stat().st_mtime, reverse=True):
        tid = pid_f.stem
        label_f = BG_TASK_DIR / f"{tid}.label"
        exit_f = BG_TASK_DIR / f"{tid}.exit"
        label = label_f.read_text(errors="replace").strip() if label_f.is_file() else "?"
        if exit_f.is_file():
            st = f"done(exit={exit_f.read_text(errors='replace').strip()})"
        else:
            try:
                os.kill(int(pid_f.read_text().strip()), 0)
                st = "running"
            except Exception:
                st = "failed"
        rows.append(f"{tid}  {st:<14} {label[:60]}")
    return "\n".join(rows) or "(no background tasks)"

# ─── TTY state ───────────────────────────────────────────────────────────────
_ORIG_TTY = None
if termios and sys.stdin.isatty():
    try:
        _ORIG_TTY = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        _ORIG_TTY = None

def ensure_cooked():
    if _ORIG_TTY is None:
        return
    try:
        fd = sys.stdin.fileno()
        termios.tcsetattr(fd, termios.TCSANOW, _ORIG_TTY)
        termios.tcflush(fd, termios.TCIFLUSH)
    except Exception:
        pass

SYSTEM_BASE = f"""You are Kimi, a full-stack coding agent working in {WORKDIR}.
You can read/write/edit files, run shell commands, and search the codebase.
Front-end, back-end, scripts, infra — you handle all of it.

Workflow:
1. For ANY task with 2+ steps: FIRST call todo_write with the complete plan
   (mark steps in_progress → completed as you go). This keeps you on track
   and shows the user live progress.
2. Briefly state your plan (2-4 bullets) before using tools.
3. Use tools to inspect before editing — never guess file contents.
4. Prefer edit_file over rewriting whole files.
5. After changes, run/verify when possible (tests, syntax check, build).
6. Final answer: concise summary of what changed and why, with file paths.

Long-running commands (installs, builds, training, big downloads): use
run_shell_bg — it returns a taskId IMMEDIATELY and keeps running while you
work. Poll with task_status(taskId). run_shell auto-converts heavy commands
(installs/clones) to background tasks — never let a long command eat the
timeout. NEVER busy-wait; do other work and check back.

Rules: be precise, minimal diffs, no placeholder code, quote commands you run.
Efficiency: read only the files you need. If a task requires editing 1-2 files, do NOT read the entire project. After 5+ tool calls, start writing your answer.
Images: NEVER read_file an image/binary — screenshots the user pastes are attached automatically as vision input. If you need to inspect an image the user did NOT attach, say so and ask for the path.

Infrastructure access (use run_shell):
- Staging server: ssh -i /Users/adamneto/jpgadmin/jpgadmin-access-key.pem -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@47.245.83.66
- Remote commands: ssh -i /Users/adamneto/jpgadmin/jpgadmin-access-key.pem -o StrictHostKeyChecking=no root@47.245.83.66 "command here"
- For DB access: SSH in, find .env or config files for credentials, then use mysql/psql CLI.
- Always use non-interactive SSH (no -t flag, single commands).
- SCP: scp -i /Users/adamneto/jpgadmin/jpgadmin-access-key.pem -o StrictHostKeyChecking=no <local> root@47.245.83.66:<remote>"""

def build_system() -> str:
    """System prompt with the live todo list and the memory ledger appended.
    Rebuilt EVERY turn so the model always sees current state — mirrors the
    archetype agent's prompt injection of SessionTodos + SessionMemory."""
    parts = [SYSTEM_BASE]
    todos = render_todos()
    if TODOS:
        parts.append("\nCURRENT TODO LIST (yours — keep it updated via todo_write):\n" + todos)
    ledger = mem_ledger()
    if ledger:
        parts.append("\nMEMORY (what you already did in recent turns — do NOT redo):\n" + ledger)
    return "\n".join(parts)

# ─── Tools schema ────────────────────────────────────────────────────────────
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file with line numbers. Never use on images/binaries.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file (creates parent dirs).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": "Replace a unique string in a file. Fails if old_string is not found exactly once.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"}, "new_string": {"type": "string"}},
            "required": ["path", "old_string", "new_string"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List directory tree.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "depth": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "run_shell",
        "description": "Run a shell command in the workspace.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "integer"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "grep",
        "description": "Search file contents for a regex pattern.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "find_files",
        "description": "Find files by glob pattern, e.g. **/*.py",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "todo_write",
        "description": "Persist and update the task list for this session. Send the COMPLETE updated todo list every time — never a partial patch. Use for ANY task with 2+ steps: create the list before starting work, mark a task in_progress when you begin it, completed when done. This keeps you on track across long multi-step builds and shows the user live progress.",
        "parameters": {"type": "object", "properties": {
            "tasks": {"type": "array", "items": {"type": "object", "properties": {
                "content": {"type": "string", "description": "What needs to be done"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                "activeForm": {"type": "string", "description": "Present-progressive label, e.g. 'Running tests'"}},
                "required": ["content", "status"]}}},
            "required": ["tasks"]}}},
    {"type": "function", "function": {
        "name": "run_shell_bg",
        "description": "Start a LONG-RUNNING command as a detached background task and return IMMEDIATELY (does not block). Use for anything that takes minutes: pip/npm installs, model training, big builds, dataset downloads. Returns a taskId — then call task_status to check progress (log tail + running/done/failed). The task keeps running while you and the user keep talking.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
            "label": {"type": "string", "description": "Short human label, e.g. 'pip install torch'"}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "task_status",
        "description": "Check the status of a background task started with run_shell_bg. Returns running/done/failed, the PID, and the last lines of its log. Call whenever the user asks 'is it done yet?' or between other work — never busy-wait.",
        "parameters": {"type": "object", "properties": {
            "taskId": {"type": "string"},
            "lines": {"type": "integer", "description": "Log lines to show from the end (default 15)"}},
            "required": ["taskId"]}}},
]

# ─── Tool implementations ────────────────────────────────────────────────────
def _p(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (WORKDIR / p)

def _is_binary(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            return b"\x00" in f.read(8192)
    except Exception:
        return False

def t_read_file(path, start_line=None, end_line=None):
    p = _p(path)
    if not p.is_file(): return f"ERROR: {p} not found"
    if p.suffix.lower() in IMG_EXTS or _is_binary(p):
        return (f"ERROR: {p.name} is a binary/image file — do NOT read_file it. "
                f"Images only reach you when the user pastes the path in chat (auto-attached as vision input). "
                f"Ask the user to paste the path if you need to see it.")
    lines = p.read_text(errors="replace").splitlines()
    s, e = (start_line or 1), (end_line or min(len(lines), 2000))
    chunk = lines[s-1:e]
    out = "\n".join(f"{i+s:>5} | {l}" for i, l in enumerate(chunk))
    if e < len(lines): out += f"\n... ({len(lines)-e} more lines)"
    return out or "(empty file)"

def _diff(old, new, path):
    d = difflib.unified_diff(old.splitlines(), new.splitlines(),
                             f"a/{path}", f"b/{path}", lineterm="", n=2)
    return "\n".join(list(d)[:200])

def t_write_file(path, content):
    p = _p(path); p.parent.mkdir(parents=True, exist_ok=True)
    old = p.read_text(errors="replace") if p.exists() else ""
    p.write_text(content)
    d = _diff(old, content, path)
    return f"Wrote {p} ({len(content)} bytes)\n{d}" if d else f"Wrote {p}"

def t_edit_file(path, old_string, new_string):
    p = _p(path)
    if not p.is_file(): return f"ERROR: {p} not found"
    text = p.read_text(errors="replace")
    n = text.count(old_string)
    if n != 1: return f"ERROR: old_string found {n} times (need exactly 1)"
    p.write_text(text.replace(old_string, new_string, 1))
    return f"Edited {p}\n{_diff(text, p.read_text(), path)}"

SKIP = {".git", "node_modules", "__pycache__", ".venv", "dist", ".next"}
def t_list_dir(path=".", depth=2):
    root, out, count = _p(path), [], [0]
    def walk(d, lvl, prefix=""):
        if lvl > (depth or 2) or count[0] > 300: return
        try: entries = sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name))
        except PermissionError: return
        for e in entries:
            if e.name in SKIP or (e.name.startswith(".") and e.name != ".env"): continue
            count[0] += 1
            out.append(f"{prefix}{'📁' if e.is_dir() else '📄'} {e.name}")
            if e.is_dir(): walk(e, lvl+1, prefix + "  ")
    walk(root, 1)
    return "\n".join(out) or "(empty)"

_CURRENT_PROC = [None]          # live subprocess (so ESC can kill it)

def _kill_proc(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try: proc.kill()
        except Exception: pass

def t_run_shell(command, timeout=None):
    if CONFIRM_SHELL and console.input(f"[yellow]run?[/] {command} [y/N] ").lower() != "y":
        return "Skipped by user."
    # Heavy commands (installs/clones/big builds) never block the turn —
    # auto-convert to a background task so the timeout can't SIGKILL a
    # multi-GB install mid-extract (same rationale as the archetype agent).
    if HEAVY_CMD_RE.search(command):
        info = bg_start(command, label="auto-bg: " + command[:50])
        return (f"BACKGROUND TASK STARTED (auto-converted — this looks like an "
                f"install/download/clone/build and must not block the turn)\n"
                f"taskId: {info['taskId']}\nlabel: {info['label']}\n"
                f"Poll with task_status(taskId=\"{info['taskId']}\") — do NOT "
                f"busy-wait; continue other work and check back.")
    proc = None
    try:
        proc = subprocess.Popen(command, shell=True, cwd=WORKDIR,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        _CURRENT_PROC[0] = proc
        out, _ = proc.communicate(timeout=timeout or SHELL_TIMEOUT)
        out = (out or "").strip()
        if len(out) > 8000: out = out[:8000] + f"\n... ({len(out)-8000} chars truncated)"
        return f"exit={proc.returncode}\n{out}" or f"exit={proc.returncode} (no output)"
    except subprocess.TimeoutExpired:
        if proc is not None: _kill_proc(proc)
        return f"ERROR: timed out after {timeout or SHELL_TIMEOUT}s"
    finally:
        _CURRENT_PROC[0] = None

def t_grep(pattern, path="."):
    rg = subprocess.run(["which", "rg"], capture_output=True).returncode == 0
    cmd = ["rg", "-n", "--max-count=50", pattern, str(_p(path))] if rg else \
          ["grep", "-rnE", "--", pattern, str(_p(path))]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKDIR)
    out = (r.stdout or r.stderr).strip()
    return out[:6000] if out else "(no matches)"

def t_find_files(pattern, path="."):
    hits = [str(p.relative_to(WORKDIR)) for p in _p(path).rglob(pattern)
            if not any(s in p.parts for s in SKIP)]
    return "\n".join(hits[:200]) or "(none)"

def t_todo_write(tasks):
    global TODOS
    TODOS[:] = normalize_todos(tasks)
    print_todos_panel()
    return f"ok — {len(TODOS)} task(s) saved"

def t_run_shell_bg(command, label=None):
    info = bg_start(command, label=label or "")
    return (f"taskId: {info['taskId']}\nlabel: {info['label']}\n"
            f"log: {info['log']}\nstatus: started (running detached — poll with "
            f"task_status(taskId=\"{info['taskId']}\"))")

def t_task_status(taskId, lines=None):
    return bg_status(taskId, int(lines) if lines else 15)

DISPATCH = {"read_file": t_read_file, "write_file": t_write_file,
            "edit_file": t_edit_file, "list_dir": t_list_dir,
            "run_shell": t_run_shell, "grep": t_grep, "find_files": t_find_files,
            "todo_write": t_todo_write, "run_shell_bg": t_run_shell_bg,
            "task_status": t_task_status}

# ─── History hygiene (the fix for "resolvable tool name" errors) ─────────────
def sanitize_history(messages):
    """Rebuild history so every assistant tool_call is followed by exactly its
    tool results (with `name`, same order), and no orphan tool messages exist.
    Kimi K3's gateway rejects anything else — this makes that impossible."""
    out, fixes, i = [], 0, 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            tcs = m["tool_calls"]
            results, j = {}, i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                results[messages[j].get("tool_call_id")] = messages[j]
                j += 1
            ordered = list(results.values())
            new_tools = []
            for k, tc in enumerate(tcs):
                nm = tc.get("function", {}).get("name") or "tool"
                tm = results.get(tc.get("id"))
                if tm is None and k < len(ordered):
                    tm = ordered[k]                      # order fallback
                if tm is None:
                    fixes += 1
                content = (tm or {}).get("content") or "(no result — interrupted)"
                new_tools.append({"role": "tool", "tool_call_id": tc.get("id"),
                                  "name": nm, "content": str(content)})
            out.append(m); out.extend(new_tools)
            i = j
        elif m.get("role") == "tool":
            fixes += 1                                    # orphan — drop
            i += 1
        else:
            out.append(m); i += 1
    messages[:] = out
    return fixes

def rollback_turn(messages, keep_user=True):
    """Rewind everything added since the last user message. Never leaves a
    dangling tool_call or orphan tool message behind. Logged so a resumed
    session applies the same rewind."""
    while len(messages) > 1 and messages[-1].get("role") != "user":
        messages.pop()
    if keep_user and len(messages) > 1 and messages[-1].get("role") == "user":
        messages.append({"role": "assistant",
                         "content": "(turn cancelled — partial tool work discarded; history is clean)"})
    elif not keep_user and len(messages) > 1 and messages[-1].get("role") == "user":
        messages.pop()
    log_event("rewind", len(messages) - 1)   # minus system
    if messages and messages[-1].get("role") == "assistant":
        log_msg(messages[-1])

# ─── Images ──────────────────────────────────────────────────────────────────
def encode_image(path: Path):
    pth = path
    if pth.suffix.lower() in (".heic", ".heif"):
        out = pth.with_suffix(".png")
        subprocess.run(["sips", "-s", "format", "png", str(pth), "--out", str(out)],
                       capture_output=True)
        if out.is_file():
            pth = out
    if pth.is_file() and pth.stat().st_size > 4_000_000:
        small = pth.with_name(pth.stem + "_kimi_small" + pth.suffix)
        subprocess.run(["sips", "-Z", "1568", str(pth), "--out", str(small)],
                       capture_output=True)
        if small.is_file():
            pth = small
    mime = mimetypes.guess_type(str(pth))[0] or "image/png"
    b64 = base64.b64encode(pth.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s)

_IMG_RE = r"(?:png|jpe?g|gif|webp|bmp|heic|heif)"

def extract_images(line):
    """Return (image_paths, remaining_text). Handles:
    - quoted paths            "…/my shot.png"
    - shell-escaped paths     …/Screenshot\\ 2026-08-23\\ at\\ 11.14.24.png
    - bare paths with spaces  …/Screenshot 2026-08-23 at 11.14.24.png
    """
    imgs, spans = [], []

    # 1) quoted
    for m in re.finditer(r'"([^"]+)"|\'([^\']+)\'', line):
        raw = _unescape((m.group(1) or m.group(2)).strip())
        pp = Path(raw).expanduser()
        if pp.suffix.lower() in IMG_EXTS and pp.is_file():
            imgs.append(pp); spans.append(m.span())

    # 2) shell-escaped tokens (backslash-escaped spaces count as one token)
    for m in re.finditer(r"(?:\\.|[^\s\\])+\." + _IMG_RE + r"\b", line, re.I):
        if any(m.start() >= s and m.end() <= e for s, e in spans):
            continue
        pp = Path(_unescape(m.group(0))).expanduser()
        if pp.is_file() and pp.suffix.lower() in IMG_EXTS:
            imgs.append(pp); spans.append(m.span())

    # 3) bare unquoted paths with real spaces (leftward expansion, loop-safe)
    for m in re.finditer(r"\S+\." + _IMG_RE + r"\b", line, re.I):
        if any(m.start() >= s and m.end() <= e for s, e in spans):
            continue
        end, best = m.end(), None
        i = m.start()
        while i > 0:
            pp = Path(_unescape(line[i:end].strip())).expanduser()
            if pp.is_file() and pp.suffix.lower() in IMG_EXTS:
                best = (i, end, pp)
            j = i
            while j > 0 and line[j-1] == " ":
                j -= 1
            while j > 0 and line[j-1] != " ":
                j -= 1
            if j == i:
                break
            i = j
        if best:
            s, e, pp = best
            imgs.append(pp); spans.append((s, e))

    text = line
    for s, e in sorted(spans, reverse=True):
        text = text[:s] + " " + text[e:]
    deduped = list(dict.fromkeys(imgs))
    return deduped, " ".join(text.split())

# ─── Key input: lone ESC vs arrow-key escape sequences ───────────────────────
def _read_key(timeout=0.0):
    """Read one keypress from stdin (already in cbreak mode).
    Returns 'esc', 'up', 'down', 'left', 'right', a decoded char, or None.
    Arrow/function keys send ESC[+X — the whole sequence is drained here so
    it can NEVER be mistaken for a lone ESC or leak into later input."""
    if not sys.stdin.isatty():
        return None
    try:
        if not select.select([sys.stdin], [], [], timeout)[0]:
            return None
        b = os.read(sys.stdin.fileno(), 1)
        if not b:
            return None
        if b != b"\x1b":
            return b.decode(errors="ignore")
        # ESC byte — check for a following sequence (arrow keys etc.)
        seq = b""
        while len(seq) < 2 and select.select([sys.stdin], [], [], 0.02)[0]:
            chunk = os.read(sys.stdin.fileno(), 1)
            if not chunk:
                break
            seq += chunk
        if seq == b"[A": return "up"
        if seq == b"[B": return "down"
        if seq == b"[C": return "right"
        if seq == b"[D": return "left"
        if seq:
            return None          # some other CSI sequence — swallow it
        return "esc"             # genuinely alone → real ESC keypress
    except Exception:
        return None

def _esc_hit():
    return _read_key() == "esc"

# ─── Global cancel watcher (ESC anywhere in a turn) ────────────────────────────
# Problem: ESC was only polled at specific points (connect, stream, tool-run).
# Anywhere else — rendering, between rounds, waiting on the queue — ESC did
# nothing. Fix: ONE watcher thread per turn, started at turn start, that reads
# stdin continuously and flips a flag. Every wait point checks the FLAG, not
# the key. Arrow keys are still drained correctly by _read_key.
_CANCEL = threading.Event()
_TURN_ACTIVE = threading.Event()

def _cancel_watcher():
    """Sole stdin reader while a turn is active. Sets _CANCEL on a lone ESC.
    Arrow keys are drained (never leak, never false-cancel). Exits when the
    turn ends so it never steals keystrokes from the input prompt."""
    while _TURN_ACTIVE.is_set() and not _CANCEL.is_set():
        k = _read_key(timeout=0.1)
        if k == "esc":
            _CANCEL.set()
            return

def start_cancel_watcher():
    _CANCEL.clear()
    _TURN_ACTIVE.set()
    if sys.stdin.isatty() and termios:
        try:
            fd = sys.stdin.fileno()
            tty.setcbreak(fd)  # KEY: make ESC readable without Enter
        except Exception:
            pass
        threading.Thread(target=_cancel_watcher, daemon=True).start()

def stop_cancel_watcher():
    _TURN_ACTIVE.clear()
    if sys.stdin.isatty() and termios:
        try:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSANOW, _ORIG_TTY)
            termios.tcflush(fd, termios.TCIFLUSH)
        except Exception:
            pass

def cancel_hit() -> bool:
    return _CANCEL.is_set()

# ─── Streaming: timeout-safe, ESC-safe ───────────────────────────────────────
class StreamTimeout(Exception):
    """API stopped sending data."""

class CancelledTurn(Exception):
    """ESC pressed at any point in the turn."""

class _CancelledStream(Exception):
    pass

class cbreak:
    """Context manager: put stdin in cbreak mode, restore on exit."""
    def __enter__(self):
        self.fd, self.old = None, None
        if termios and sys.stdin.isatty():
            try:
                self.fd = sys.stdin.fileno()
                self.old = termios.tcgetattr(self.fd)
                tty.setcbreak(self.fd)
            except Exception:
                self.old = None
        return self
    def __exit__(self, *exc):
        if self.old is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, _ORIG_TTY or self.old)
                termios.tcflush(self.fd, termios.TCIFLUSH)
            except Exception:
                pass

def stream_chunks(stream, idle_timeout=90):
    """Yield chunks via a background reader so ESC + idle-timeout stay responsive."""
    q = queue.Queue()
    def _reader():
        try:
            for c in stream:
                q.put(c)
        except Exception as e:
            q.put(e)
        finally:
            q.put(None)
    threading.Thread(target=_reader, daemon=True).start()
    idle = 0.0
    while True:
        if cancel_hit():
            raise _CancelledStream()
        try:
            item = q.get(timeout=0.2)
        except queue.Empty:
            idle += 0.2
            if idle >= idle_timeout:
                raise StreamTimeout(f"no data from API for {idle_timeout}s")
            continue
        idle = 0.0
        if item is None:
            return
        if isinstance(item, Exception):
            raise item
        yield item

def tail_text(s, max_chars):
    if max_chars is None or len(s) <= max_chars:
        return s
    return "… " + s[-max_chars:]

def print_usage(usage):
    if usage:
        console.print(f"[dim]  · {getattr(usage,'completion_tokens','?')} out / "
                      f"{getattr(usage,'prompt_tokens','?')} in tokens[/]")

def _connect(model, messages):
    """Open the stream on a worker thread; spinner + ESC while connecting."""
    slot = {}
    def _go():
        try:
            slot["s"] = client.chat.completions.create(
                model=model, messages=messages, tools=TOOLS, tool_choice="auto",
                stream=True, stream_options={"include_usage": True},
                extra_body=EXTRA_BODY)
        except Exception as e:
            slot["e"] = e
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    with console.status("[cyan]connecting to kimi… (esc to cancel)", spinner="dots"):
        while t.is_alive():
            if cancel_hit():
                raise CancelledTurn()
            t.join(timeout=0.1)
    if "e" in slot:
        raise slot["e"]
    if "s" not in slot:
        raise CancelledTurn()
    return slot["s"]

def run_tool_esc(name, args):
    """Run a tool on a worker thread; ESC kills any live shell proc and cancels."""
    slot = {}
    def _go():
        try:
            slot["r"] = DISPATCH[name](**args) if name in DISPATCH else "unknown tool"
        except Exception as e:
            slot["r"] = f"ERROR in {name}: {e}"
    t = threading.Thread(target=_go, daemon=True)
    t.start()
    while t.is_alive():
        if cancel_hit():
            if _CURRENT_PROC[0] is not None:
                _kill_proc(_CURRENT_PROC[0])
            t.join(timeout=1.5)
            raise CancelledTurn()
        t.join(timeout=0.1)
    return slot.get("r", "(no result)")

# ─── Agent loop ──────────────────────────────────────────────────────────────
def run_turn(messages, has_images=False):
    model = (VISION_MODEL or MODEL) if has_images else MODEL
    usage, step = None, [0]

    # Rebuild the system prompt EVERY turn: live todos + memory ledger are
    # injected so the model always sees current state (mirrors the archetype
    # agent's per-turn prompt assembly).
    if messages and messages[0].get("role") == "system":
        messages[0] = {"role": "system", "content": build_system()}

    fixes = sanitize_history(messages)
    if fixes:
        console.print(f"[dim]  · history repaired ({fixes} fix(es))[/]")

    start_cancel_watcher()          # ESC works ANYWHERE from here on
    try:
        _run_turn_inner(messages, model, usage, step, has_images)
    finally:
        stop_cancel_watcher()

def _run_turn_inner(messages, model, usage, step, has_images=False):
    for rnd in range(MAX_TOOL_ROUNDS):
        stream = _connect(model, messages)

        reasoning, content, tcs = "", [], {}
        cancelled = [False]

        def render():
            parts = []
            if reasoning:
                parts.append(Text(("… " if len(reasoning) > 1400 else "") + tail_text(reasoning, 1400),
                                  style="dim italic"))
            if content:
                parts.append(Text("".join(content)))
            if not parts:
                parts.append(Text("thinking…", style="dim"))
            parts.append(Text("\nesc to cancel", style="dim"))
            return Panel(Group(*parts),
                         title=f"[bold cyan]{model}[/] · step {step[0]+1}", box=box.ROUNDED,
                         border_style="cyan")

        try:
            with Live(render(), console=console, refresh_per_second=8) as live:
                for chunk in stream_chunks(stream):
                    if chunk.usage: usage = chunk.usage
                    if not chunk.choices: continue
                    d = chunk.choices[0].delta
                    rc = getattr(d, "reasoning_content", None)
                    if rc: reasoning += rc
                    if d.content: content.append(d.content)
                    if d.tool_calls:
                        for tc in d.tool_calls:
                            s = tcs.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                            if tc.id: s["id"] += tc.id
                            if tc.function.name: s["name"] += tc.function.name
                            if tc.function.arguments: s["args"] += tc.function.arguments
                    if rc or d.content:
                        live.update(render())
        except _CancelledStream:
            cancelled[0] = True
        finally:
            try: stream.close()
            except Exception: pass

        text = "".join(content)

        if cancelled[0]:
            step[0] += 1
            console.print(Panel(Markdown(text) if text else Text("(no output)"),
                                title="[bold yellow]cancelled (esc)", box=box.ROUNDED,
                                border_style="yellow"))
            if tcs:
                raise CancelledTurn()          # partial tool call → rewind whole turn
            messages.append({"role": "assistant", "content": text or "(cancelled by user)"})
            log_msg(messages[-1])
            return

        if not tcs:
            print_usage(usage)
            messages.append({"role": "assistant", "content": text})
            log_msg(messages[-1])
            mem_record("answer", "Answer", result=text[:600])
            return

        # tool calls — execute in order, each result carries `name`
        ordered = [tcs[i] for i in sorted(tcs)]
        messages.append({"role": "assistant", "content": text or None,
                         "tool_calls": [{"id": t["id"], "type": "function",
                                         "function": {"name": t["name"], "arguments": t["args"]}}
                                        for t in ordered]})
        log_msg(messages[-1])
        for t in ordered:
            step[0] += 1
            try: args = json.loads(t["args"] or "{}")
            except json.JSONDecodeError: args = {"_raw": t["args"]}
            preview = args.get("command") or args.get("path") or args.get("pattern") or ""
            console.print(Panel(f"[bold]{t['name']}[/] [dim]{preview}[/]",
                                title=f"[yellow]⚙ step {step[0]}", box=box.ROUNDED,
                                border_style="yellow"))
            with console.status(f"[yellow]running {t['name']}… (esc to cancel)", spinner="dots"):
                result = run_tool_esc(t["name"], args)
            show = result if len(result) < 1500 else result[:1500] + "\n…(truncated)"
            lexer = "diff" if "\n@@" in show or show.startswith(("Edited", "Wrote")) else "text"
            console.print(Panel(Syntax(show, lexer, word_wrap=True, theme="monokai")
                                if lexer == "diff" else Text(show, style="dim"),
                                box=box.SIMPLE, border_style="dim"))
            messages.append({"role": "tool", "tool_call_id": t["id"],
                             "name": t["name"], "content": result})
            log_msg(messages[-1])
            # Memory ledger: record each tool step (args + result summary)
            mem_record("tool", t["name"],
                       detail=preview if isinstance(preview, str) else "",
                       result=(result or "")[:300],
                       ok=not (result or "").startswith(("ERROR", "exit=1", "exit=2")))

    # MAX_TOOL_ROUNDS exhausted — synthesize any missing results, then summarize
    sanitize_history(messages)
    console.print("[yellow]⚠ hit tool-round limit — asking for summary… (esc to skip)[/]")
    messages.append({"role": "assistant", "content": "(tool round limit reached — summarizing)"})
    log_msg(messages[-1])
    try:
        stream = _connect(model, messages)
        final_text = []
        for chunk in stream_chunks(stream):
            if chunk.choices and chunk.choices[0].delta.content:
                final_text.append(chunk.choices[0].delta.content)
        summary = "".join(final_text)
        if summary:
            console.print(Panel(Markdown(summary), title="[bold green]summary",
                                box=box.ROUNDED, border_style="green"))
            messages.append({"role": "assistant", "content": summary})
            log_msg(messages[-1])
            mem_record("answer", "Answer", result=summary[:600])
    except (CancelledTurn, _CancelledStream):
        console.print("[dim]summary skipped (esc)[/]")
    except Exception as e:
        console.print(f"[dim]summary call failed: {e}[/]")

# ─── Model failover ──────────────────────────────────────────────────────────
_FAILOVER_NEEDLES = ("balance", "no suitable", "insufficient", "quota",
                     "does not exist", "model_not_found", "not found",
                     "404", "429", "rate limit", "too many requests",
                     "throttling", "resourceexhausted", "resource exhausted",
                     "overloaded", "capacity", "unavailable",
                     "forbidden", "not enabled", "permission denied")

def _is_failover_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(n in msg for n in _FAILOVER_NEEDLES)

def _chain():
    """Current model first, then the rest of the failover chain."""
    return [MODEL] + [m for m in FAILOVER_CHAIN if m != MODEL]

def _next_failover(has_images=False, exclude=()):
    for m in _chain():
        if m in DEAD_MODELS or m in exclude:
            continue
        if has_images and not VISION_MODEL and m in NO_VISION:
            continue
        return m
    return None

def run_turn_with_failover(messages, has_images=False):
    """run_turn + automatic failover. Dead models are skipped for the rest of
    the session; history is sanitized before each retry so partial tool work
    carries over cleanly to the next model."""
    global MODEL
    if MODEL in DEAD_MODELS or (has_images and not VISION_MODEL and MODEL in NO_VISION):
        nxt = _next_failover(has_images)
        if nxt:
            console.print(f"[dim]auto-switch: {MODEL} → {nxt} (previous model unavailable)[/]")
            MODEL = nxt
    tried = set()
    while True:
        try:
            run_turn(messages, has_images=has_images)
            return
        except (CancelledTurn, StreamTimeout):
            raise
        except Exception as e:
            if not _is_failover_error(e):
                raise
            DEAD_MODELS.add(MODEL)
            if has_images and not VISION_MODEL:
                NO_VISION.add(MODEL)
            nxt = _next_failover(has_images, exclude=tried | {MODEL})
            if nxt is None:
                raise
            tried.add(MODEL)
            sanitize_history(messages)
            short = str(e).strip().splitlines()[0][:120]
            console.print(f"[yellow]⚠ {MODEL}: {short}[/]")
            console.print(f"[yellow]→ failing over to [bold]{nxt}[/bold][/]")
            MODEL = nxt

# ─── Slash commands (live-filtered menu, like the web app's picker) ───────────
SLASH_COMMANDS = [
    ("/model",   "switch models (arrow-key picker) · /model list"),
    ("/todos",   "show the agent's current todo list"),
    ("/tasks",   "list background tasks (running/done/failed)"),
    ("/bg",      "run a command in the background: /bg <command>"),
    ("/memory",  "show the persistent memory ledger"),
    ("/memory clear", "wipe the memory ledger"),
    ("/img",     "attach an image: /img <path>"),
    ("/cd",      "change working directory: /cd <dir>"),
    ("/reset",   "clear conversation history"),
    ("/doctor",  "check + repair message history"),
    ("/exit",    "quit"),
]

def _slash_matches(frag: str) -> list:
    """Commands matching the typed fragment — prefix first, then substring."""
    frag = frag.lower()
    exact = [c for c in SLASH_COMMANDS if c[0].startswith(frag)]
    extra = [c for c in SLASH_COMMANDS if frag in c[0] and c not in exact]
    return exact + extra

def _print_menu(matches, sel):
    """Render the filtered menu below the prompt (dim, selected row bright)."""
    if not matches:
        return
    rows = []
    for i, (cmd, desc) in enumerate(matches[:8]):
        cur = "▶ " if i == sel else "  "
        style = "bold cyan" if i == sel else "dim"
        rows.append(f"[{style}]{cur}{cmd:<16} {desc}[/]")
    console.print("\n".join(rows), highlight=False)

def _slash_input(prompt_str):
    """Input with a live-filtered slash menu. Uses raw stdout writes so the
    prompt line is overwritten in-place (no spam)."""
    if not sys.stdin.isatty() or termios is None:
        try:
            return input(prompt_str)
        except (EOFError, KeyboardInterrupt):
            return None

    buf, sel, menu_open = "", 0, False
    menu_lines = 0

    def _redraw():
        nonlocal menu_lines
        if menu_lines > 0:
            sys.stdout.write(f"\033[{menu_lines}A")
            menu_lines = 0
        sys.stdout.write(f"\r\033[2K\033[1;36m{prompt_str}\033[0m{buf}")
        sys.stdout.flush()
        if menu_open and buf.startswith("/"):
            matches = _slash_matches(buf)
            if matches:
                sys.stdout.write("\n")
                shown = matches[:8]
                for i, (cmd, desc) in enumerate(shown):
                    cur = "\u25b6 " if i == sel else "  "
                    if i == sel:
                        sys.stdout.write(f"\033[1;36m{cur}{cmd:<16} {desc}\033[0m\n")
                    else:
                        sys.stdout.write(f"\033[2m{cur}{cmd:<16} {desc}\033[0m\n")
                menu_lines = len(shown) + 1
                sys.stdout.flush()

    with cbreak():
        _redraw()
        while True:
            k = _read_key(timeout=3600)
            if k is None:
                continue
            if k == "esc":
                if menu_open:
                    menu_open, sel = False, 0
                    _redraw()
                    continue
                sys.stdout.write("\r\033[2K\n")
                sys.stdout.flush()
                return None
            if k in ("up", "down"):
                if menu_open:
                    matches = _slash_matches(buf)
                    if matches:
                        sel = (sel + (1 if k == "down" else -1)) % len(matches)
                        _redraw()
                continue
            if k == "\t" and menu_open:
                matches = _slash_matches(buf)
                if matches:
                    buf = matches[sel][0]
                    sel = 0
                    _redraw()
                continue
            if k in ("\r", "\n"):
                if menu_lines > 0:
                    sys.stdout.write(f"\033[{menu_lines}A")
                    menu_lines = 0
                sys.stdout.write(f"\r\033[2K\033[1;36m{prompt_str}\033[0m{buf}\n")
                sys.stdout.flush()
                if menu_open:
                    matches = _slash_matches(buf)
                    if matches and matches[sel][0] != buf:
                        return matches[sel][0]
                return buf
            if k in ("\x7f", "\b"):
                buf = buf[:-1]
                menu_open = buf.startswith("/")
                sel = 0
                _redraw()
                continue
            if isinstance(k, str) and k.isprintable() and len(k) == 1:
                buf += k
                menu_open = buf.startswith("/")
                sel = 0
                _redraw()
                continue
    return None


# ─── Smart input (multi-line paste aware) ────────────────────────────────────
def smart_input(prompt_str):
    # Slash-command mode: live-filtered menu below the prompt
    line = _slash_input(prompt_str)
    if line is None:
        return None
    if not line.startswith("/"):
        # multi-line paste detection (unchanged behavior)
        lines = [line]
        time.sleep(0.05)
        while True:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    break
                more = sys.stdin.readline()
                if not more:
                    break
                lines.append(more.rstrip("\n"))
            except Exception:
                break
        if len(lines) > 1:
            combined = "\n".join(lines)
            console.print(f"  [dim]{{Pasted {len(lines)} lines}}[/]")
            return combined
    return line


def _model_picker(current_model):
    """Interactive model picker — arrow keys + enter. Returns a model id or None.
    Uses the MODELS registry (chat models only, aliases shown) and _read_key()
    so arrow-key escape sequences are parsed exactly like everywhere else in
    the REPL — no buffered stdin, no ESC/arrow ambiguity."""
    if not termios or not sys.stdin.isatty():
        try:
            return resolve_model(input("Model name: ").strip()) or None
        except (EOFError, KeyboardInterrupt):
            return None
    entries = list(MODELS.items())            # [(alias, model_id), ...]
    sel = next((i for i, (_, mid) in enumerate(entries) if mid == current_model), 0)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    nlines = len(entries) + 2                 # header + entries + footer

    def render():
        sys.stdout.write("\r\033[2K\033[1m  Select model:\033[0m\n")
        for j, (alias, mid) in enumerate(entries):
            arrow = "▶ " if j == sel else "  "
            sty = "\033[1;36m" if j == sel else "\033[2m"
            tags = ""
            if mid in DEAD_MODELS: tags += " ·dead"
            if mid in NO_VISION:   tags += " ·no-vision"
            sys.stdout.write(f"\r\033[2K  {arrow}{sty}{alias:<9} {mid}{tags}\033[0m\n")
        sys.stdout.write("\r\033[2K  \033[2m↑↓ move · enter select · esc/q cancel\033[0m\n")
        sys.stdout.flush()

    try:
        tty.setcbreak(fd)
        sys.stdout.write("\n" * nlines)       # make room below the prompt, then paint
        render()
        while True:
            k = _read_key(timeout=None)       # block until a real keypress
            if k in ("up", "k"):
                sel = max(0, sel - 1)
            elif k in ("down", "j"):
                sel = min(len(entries) - 1, sel + 1)
            elif k in ("\r", "\n"):
                return entries[sel][1]
            elif k in ("esc", "q"):
                return None
            else:
                continue                      # swallow anything else
            sys.stdout.write(f"\033[{nlines}A")  # rewind to header, repaint
            render()
    except (EOFError, KeyboardInterrupt):
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ─── REPL ────────────────────────────────────────────────────────────────────
BANNER = f"""[bold cyan]Kimi — Full-Stack Coding Agent[/] [dim]({MODEL})[/]
[dim]workspace: {WORKDIR}
commands: type / to see them all (live-filtered menu) · /model [alias] · /todos · /tasks
          /bg <cmd> run in background · /img <path> attach · /reset · /cd <dir> · /doctor · /exit
failover: auto-switch on BalanceError/unavailable → {' → '.join(FAILOVER_CHAIN)}
agent: todo lists + persistent memory (auto) · long commands run as background tasks
tips: esc cancels ANYTHING mid-turn (connecting, streaming, tools) · paste a screenshot
path any way you like — quoted, \\-escaped, or bare with spaces · /doctor checks history[/]"""

def main():
    global WORKDIR, MODEL
    messages = [{"role": "system", "content": build_system()}]
    pending_imgs: list = []
    console.print(Panel(BANNER, box=box.DOUBLE, border_style="cyan"))

    while True:
        try:
            ensure_cooked()
            line = (smart_input("\n\033[1;36myou › \033[0m") or "").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/]"); break
        if not line: continue

        if line in ("/exit", "/quit"): break
        if line == "/reset":
            messages = [{"role": "system", "content": build_system()}]; pending_imgs.clear()
            console.print("[dim]history cleared[/]"); continue
        if line == "/todos":
            print_todos_panel() if TODOS else console.print("[dim]no active todos[/]")
            continue
        if line == "/tasks":
            console.print(Panel(bg_list(), title="[bold]background tasks",
                                box=box.ROUNDED, border_style="cyan"))
            continue
        if line.startswith("/bg "):
            cmd = line[4:].strip()
            if not cmd:
                console.print("[red]usage: /bg <command>[/]")
            else:
                info = bg_start(cmd)
                console.print(f"[dim]started {info['taskId']} — {info['label']}[/]")
                console.print(f"[dim]log: {info['log']} · check with /tasks[/]")
            continue
        if line == "/memory":
            ledger = mem_ledger(max_turns=10, full_turns=10)
            console.print(Panel(ledger or "(memory empty)", title="[bold]memory ledger",
                                box=box.ROUNDED, border_style="cyan"))
            continue
        if line == "/memory clear":
            mem_clear()
            console.print("[dim]memory cleared[/]")
            continue
        if line == "/doctor":
            n = sanitize_history(messages)
            roles = {}
            for m in messages: roles[m.get("role","?")] = roles.get(m.get("role","?"),0)+1
            console.print(f"[dim]{len(messages)} messages {roles} · {n} fix(es) applied · history valid[/]")
            continue
        if line.startswith("/cd "):
            t = _p(line[4:].strip())
            if t.is_dir(): WORKDIR = t.resolve(); console.print(f"[dim]cwd → {WORKDIR}[/]")
            else: console.print("[red]no such dir[/]")
            continue
        if line == "/model" or line.startswith("/model "):
            arg = line[6:].strip()
            if not arg:
                picked = _model_picker(MODEL)
                if not picked:
                    console.print("[dim]cancelled — model unchanged[/]")
                elif picked == MODEL:
                    console.print(f"[dim]model unchanged → {MODEL}[/]")
                else:
                    MODEL = picked
                    DEAD_MODELS.discard(picked); NO_VISION.discard(picked)
                    rest = " → ".join(_chain()[1:]) or "(no fallbacks)"
                    console.print(f"[dim]model → {MODEL} · failover: {rest}[/]")
                continue
            if arg == "list":
                rows = []
                for alias, mid in MODELS.items():
                    cur = "▶" if mid == MODEL else " "
                    tag = ""
                    if mid in DEAD_MODELS: tag += " [red]dead[/]"
                    if mid in NO_VISION:   tag += " [dim]no-vision[/]"
                    rows.append(f"{cur} [cyan]{alias:<9}[/] {mid}{tag}")
                console.print(Panel("\n".join(rows), title="[bold]models",
                                    box=box.ROUNDED, border_style="cyan"))
                console.print("[dim]chain: " + " → ".join(_chain()) + "[/]")
                continue
            new = resolve_model(arg)
            if new not in MODELS.values():
                console.print(f"[yellow]'{arg}' is not a known alias — using as raw model id[/]")
            MODEL = new
            DEAD_MODELS.discard(new); NO_VISION.discard(new)
            rest = " → ".join(_chain()[1:]) or "(no fallbacks)"
            console.print(f"[dim]model → {MODEL} · failover: {rest}[/]")
            continue
        if line.startswith("/img "):
            imgs, leftover = extract_images(line[5:].strip())
            if not imgs:
                for tok in shlex.split(line[5:]):
                    q = _p(_unescape(tok))
                    if q.is_file(): imgs.append(q)
            for q in imgs:
                pending_imgs.append(q); console.print(f"[dim]📎 {q.name}[/]")
            if not imgs: console.print(f"[red]not found: {line[5:].strip()}[/]")
            continue

        imgs, rest = extract_images(line)
        all_imgs = pending_imgs + [q for q in imgs if q not in pending_imgs]
        pending_imgs = []
        all_imgs = [q for q in all_imgs if q.is_file()]
        text = rest.strip() or (line if not imgs else "")

        if all_imgs:
            content = [{"type": "text", "text": text or "Describe this image."}]
            content += [encode_image(q) for q in all_imgs]
            console.print(f"[dim]📎 attached: {', '.join(f'{q.name} ({q.stat().st_size/1e6:.1f} MB)' for q in all_imgs)}[/]")
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": text})
        mem_record("prompt", text or "(image)", result="")   # start a new memory turn

        try:
            run_turn_with_failover(messages, has_images=bool(all_imgs))
        except CancelledTurn:
            rollback_turn(messages, keep_user=True)
            console.print("[dim]turn cancelled — history rewound cleanly[/]")
        except StreamTimeout as e:
            rollback_turn(messages, keep_user=False)
            console.print(f"[bold red]stream stalled:[/] {e} — try again")
        except Exception as e:
            rollback_turn(messages, keep_user=False)
            sanitize_history(messages)
            console.print(f"[bold red]API error:[/] {e}")
            console.print("[dim]history auto-repaired — safe to continue (or /doctor to verify)[/]")

        if len(messages) > 60:
            messages[:] = [messages[0]] + messages[-50:]
            sanitize_history(messages)

if __name__ == "__main__":
    main()


import random
_THROTTLE_NEEDLES = ("429", "rate limit", "too many requests", "throttling",
                     "resourceexhausted", "resource exhausted",
                     "overloaded", "capacity")

def _is_throttle_error(err):
    m = str(err).lower()
    return any(k in m for k in _THROTTLE_NEEDLES)

def _create_with_retry(client, kwargs, max_attempts=6):
    import time as _t
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not _is_throttle_error(e) or attempt == max_attempts - 1:
                raise
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            print(f"[throttled] retry in {wait:.1f}s ({attempt+1}/{max_attempts})")
            _t.sleep(wait)
