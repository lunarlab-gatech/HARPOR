"""Claude Opus 5.5 as the Stage B agent, through our own Messages API loop (build_plan.md note 5).

The controller starts it as the session's agent process, from the repo root, with its own
interpreter:

    python -m harpor.agent.claude_agent ROOT --sif SIF --sock-dir DIR --key-file FILE
           [--effort high] [--model claude-opus-5-5] [--shell host]

The key comes from FILE (a line ANTHROPIC_API_KEY=...) into memory only. It goes to the anthropic
client and nowhere else: not to the shell, not to the log, not to stdout. The system prompt is the
file claude/SYSTEM.md and the first user message is ROOT/session/prompt.txt, both unchanged. The
model has three tools:

  bash                         one persistent bash in the agent's container (controller.shell_command):
                               cwd /work, no network, at most 300 s per command, output cut to 30000
                               characters
  str_replace_based_edit_tool  view, create, str_replace and insert on text files under ROOT/work. It
                               runs on the host, so paths are confined to ROOT/work and no symlink is
                               followed out of it
  view_image                   a PNG or JPEG under ROOT/work, downscaled to at most MAX_IMAGE_SIDE px
                               on the long side (2576, the limit of Opus 5.5), within the size caps
                               MAX_IMAGE_B64 and IMAGE_BUDGET_BYTES

Each request is streamed with client.beta.messages.stream: adaptive thinking with summaries, the
effort given (default high; the model's own default is medium), server-side refusal fallback
("default"), and one cache breakpoint on the system prompt plus automatic caching of the
conversation. The history is append-only: response.content goes back unchanged as the assistant
turn, and system and tools never change. The one exception is the API's rule for a
fallback mid-output: before the last fallback block, only text and fallback blocks are sent back.
A turn whose stream breaks off after it opened (an error event, a dropped connection, no stop
reason), which the SDK's retries do not cover, is sent again, as is one whose tool input the SDK
could not parse. Tools run only for stop reason tool_use.

Log: ROOT/session/claude.jsonl, one JSON object per line (start, request, retry, response,
tool_result, end), appended and flushed; stdout gets a short trace. The cost estimate prices every
attempt that usage.iterations lists (a declined attempt and its fallback) at the prices of the model
that ran it; attempts that failed and were sent again are billed too but not counted ("reissued" in
the end line). Exit codes: 0 the model ended its turn or the finish helper ran, 2 refusal, 3 API
error, 130 SIGINT or SIGTERM, 1 anything else (end reason "error", then the traceback). --shell host
runs bash on the host in ROOT/work instead of Apptainer, for tests and local runs only.
"""

import argparse
import base64
import contextlib
import hashlib
import io
import itertools
import json
import os
import queue
import re
import shlex
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

import anthropic
import httpx2

from harpor.agent import tools

AGENT_DIR = Path(__file__).resolve().parent
SYSTEM_FILE = AGENT_DIR / "claude" / "SYSTEM.md"
MODEL, EFFORTS = tools.CLAUDE_MODEL, tools.EFFORTS
MAX_TOKENS = 64000
BETAS = [tools.FALLBACK_BETA]
MAX_RETRIES = 4  # the SDK's own retries of connection errors, 408, 409, 429 and 5xx
JSON_RETRIES = 2  # re-issues of a turn whose eager tool input the SDK could not parse
# Re-issues of a turn whose stream broke off after it opened (the SDK retries only the request itself),
# after RETRY_BACKOFF_S, doubled each time. RETRY_TYPES: the retryable error events (429, 500, 529).
STREAM_RETRIES, RETRY_BACKOFF_S = 3, 2.0
RETRY_TYPES = ("rate_limit_error", "api_error", "overloaded_error")
MAX_PAUSE_TURNS = 5  # pause_turn continuations in a row
COMMAND_TIMEOUT_S = 300.0
POLL_S, DRAIN_S = 0.5, 0.5  # how often a running command checks that the shell is alive; output read after it exited
MAX_OUTPUT_CHARS, HEAD_CHARS = 30000, 2000
MAX_VIEW_CHARS = 30000
MAX_IMAGE_SIDE = 2576  # the long side of Opus 5.5's high-resolution images
# A PNG whose base64 fits MAX_IMAGE_B64 after any downscaling is sent as PNG, lossless. A larger PNG,
# and a JPEG that is downscaled or does not fit, goes as JPEG at the first of JPEG_QUALITIES that fits.
MAX_IMAGE_B64, JPEG_QUALITIES = 2_500_000, (92, 80, 70)
API_IMAGE_BYTES = 3_750_000  # its base64 stays under the API's 5 MB per image
# Every image stays in the history and goes with every later request, which the API limits to 32 MB:
# the base64 of all images of a session, and their number, are capped.
IMAGE_BUDGET_BYTES, MAX_IMAGES = 24_000_000, 100
MAX_READ_BYTES = 64 << 20
LOG_OUTPUT_CHARS = 4000
# Shell output kept in memory while a command runs: the first and last bytes once it passes TRIM_AT.
KEEP_HEAD, KEEP_TAIL, TRIM_AT = 8 << 10, 256 << 10, 1 << 20
EXIT = {"ended": 0, "refusal": 2, "api_error": 3, "interrupted": 130}
# Never passed to the host shell: every variable of the API providers (keys, tokens, base URLs), and
# BASH_ENV and ENV, which would make bash source a file even with --norc. (In Apptainer, --containall
# already gives the shell a clean environment.)
DROP_ENV, DROP_PREFIXES = ("BASH_ENV", "ENV"), ("ANTHROPIC_", "OPENAI_", "CODEX_")
PNG_MAGIC, JPEG_MAGIC = b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"
# Most specific first; the label goes into the end line.
API_ERRORS = (
    (anthropic.AuthenticationError, "the API key was rejected"),
    (anthropic.PermissionDeniedError, "the key may not use this model or feature"),
    (anthropic.NotFoundError, "model or endpoint not found"),
    (anthropic.BadRequestError, "the API rejected the request"),
    (anthropic.RequestTooLargeError, "the request is too large (the images in the history)"),
    (anthropic.RateLimitError, "rate limited, also after the SDK's retries"),
    (anthropic.InternalServerError, "API server error, also after the SDK's retries"),
    (anthropic.OverloadedError, "API overloaded, also after the SDK's retries"),
    (anthropic.APIStatusError, "the API answered with an error status"),
    (anthropic.APIConnectionError, "could not reach the API, also after the SDK's retries"),
    (anthropic.APIError, "API error"),
)
# The tools of a request.
TOOLS = [
    {"type": "bash_20250124", "name": "bash"},
    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": MAX_VIEW_CHARS},
    {"name": "view_image",
     "description": ("Look at a PNG or JPEG file under /work, for example a review sheet, a single review render, "
                     "an input photo or a mask. Call it whenever you need to see an image: bash and the text editor "
                     "cannot show images. The path is /work/... or relative to /work. Images larger than "
                     f"{MAX_IMAGE_SIDE} px on the long side are shown downscaled. A PNG is sent lossless when it "
                     f"fits {MAX_IMAGE_B64 / 1e6:g} MB as base64 (about {MAX_IMAGE_B64 * 3 / 4e6:.1f} MB of file); "
                     "larger images are sent as JPEG. Every image stays in the conversation, so a session can show "
                     f"at most {MAX_IMAGES} images and {IMAGE_BUDGET_BYTES // 1_000_000} MB of them in all."),
     "eager_input_streaming": True,
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"],
                      "additionalProperties": False}},
]


class ToolError(Exception):
    """A tool call that failed; the message goes back to the model as an is_error result."""


class Interrupted(BaseException):
    """SIGINT or SIGTERM arrived. A BaseException, so no tool's error handling swallows it."""


class TurnFailed(Exception):
    """A turn that failed also after its re-issues; failed: the attempts of it that were sent."""

    def __init__(self, message, failed):
        super().__init__(message)
        self.failed = failed


class BadToolJSON(TurnFailed):
    """The SDK could not parse a turn's tool input, also after JSON_RETRIES re-issues."""


class StreamFailed(TurnFailed):
    """A turn's stream broke off after it opened, also after STREAM_RETRIES re-issues."""


def redact(text, secret):
    return text.replace(secret, "[redacted]") if secret else text


def read_key(key_file):
    """The value of the ANTHROPIC_API_KEY= line in key_file, or None (also when the file cannot be read)."""
    try:
        return next(iter(tools.env_values(key_file, ("ANTHROPIC_API_KEY",))), None) or None
    except OSError:
        return None


class Log:
    """ROOT/session/claude.jsonl: one JSON object per line, appended and flushed. Every line passes
    through redact first."""

    def __init__(self, path, secret=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f, self.secret = open(path, "a", encoding="utf-8"), secret

    def write(self, kind, **fields):
        line = json.dumps({"t": round(time.time(), 3), "type": kind, **fields}, ensure_ascii=False, default=str)
        self.f.write(redact(line, self.secret) + "\n")  # one write, so a signal cannot split a line
        self.f.flush()

    def close(self):
        self.f.close()


# ---------------------------------------------------------------------------------------------- shell


def _children(proc="/proc"):
    """{ppid: [pid, ...]} of every process: from /proc on Linux, from ps elsewhere."""
    kids = {}
    if os.path.isdir(f"{proc}/self"):
        for d in os.listdir(proc):
            if d.isdigit():
                with contextlib.suppress(OSError, ValueError, IndexError):
                    # "pid (comm) state ppid ...": comm may hold spaces and parentheses
                    ppid = int(Path(f"{proc}/{d}/stat").read_text().rsplit(")", 1)[1].split()[1])
                    kids.setdefault(ppid, []).append(int(d))
        return kids
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            with contextlib.suppress(ValueError):
                pid, ppid = map(int, line.split())
                kids.setdefault(ppid, []).append(pid)
    return kids


def kill_tree(pid):
    """SIGKILL pid and everything it started. The shell shares the agent's process group, so the group
    cannot be signalled; Apptainer's processes and the commands inside are found by parent PID."""
    kids, tree, todo = _children(), [], [pid]
    while todo:
        p = todo.pop()
        tree.append(p)
        todo += kids.get(p, [])
    for p in tree:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(p, signal.SIGKILL)


def _pump(pipe, out):
    """Reader thread: the shell's output in chunks, then None at end of file. Closes the pipe itself,
    so its descriptor is never reused while a read may still be pending."""
    try:
        fd = pipe.fileno()
        while chunk := os.read(fd, 65536):
            out.put(chunk)
    except (OSError, ValueError):
        pass
    finally:
        out.put(None)
        with contextlib.suppress(OSError):
            pipe.close()


def cap_output(text, omitted=0, limit=MAX_OUTPUT_CHARS, head=HEAD_CHARS):
    """text cut to at most limit characters: the first head and the last ones, with a line that says
    how many were left out. omitted: what was already dropped from the middle while reading (counted
    in bytes)."""
    if len(text) <= limit and not omitted:
        return text
    reserve = len(f"\n[... {len(text) + omitted} characters omitted ...]\n")
    tail = max(0, min(len(text) - head, limit - head - reserve))
    n = len(text) - head - tail + omitted
    return text[:head] + f"\n[... {n} characters omitted ...]\n" + (text[len(text) - tail:] if tail else "")


def _with_code(text, code):
    if code:
        text += ("" if not text or text.endswith("\n") else "\n") + f"[exit code {code}]"
    return text or "(no output)"


class ContainerShell:
    """One persistent `bash --noprofile --norc` for the bash tool, started on first use.

    In Apptainer (controller.shell_command: no network, no key) or, with host=True, on the host with cwd
    root/work and the helper bin folder first on PATH. stderr is merged into stdout. Each command goes
    to stdin as `eval -- 'command' < /dev/null` (so a syntax error or a command that reads stdin cannot
    eat what follows) and then a printf of an end marker with a random token and $?. A reader thread
    collects the output until the marker. The shell's exit is seen from end of file or, every POLL_S,
    from the process itself (a job left in the background can hold the output open). The shell stays
    in the agent's process group (no setsid), so the controller's watchdog signals reach it; stop()
    kills it with everything it started.
    """

    def __init__(self, root, sif=None, sock_dir=None, host=False, timeout=COMMAND_TIMEOUT_S):
        self.root, self.sif, self.sock_dir, self.host, self.timeout = Path(root).resolve(), sif, sock_dir, host, timeout
        self.proc = self.out = None

    def command(self):
        if self.host:
            return ["bash", "--noprofile", "--norc"]
        from harpor.agent import controller  # here, so --shell host runs without the controller's imports

        return controller.shell_command(self.root, self.sif, self.sock_dir)

    def environment(self):
        env = {k: v for k, v in os.environ.items() if k not in DROP_ENV and not k.startswith(DROP_PREFIXES)}
        if self.host:
            env["PATH"] = f"{AGENT_DIR / 'bin'}{os.pathsep}{env.get('PATH', os.defpath)}"
            if self.sock_dir:
                env["GATOR_SOCKET"] = str(Path(self.sock_dir) / "tools.sock")
        return env

    def start(self):
        self.proc = subprocess.Popen(self.command(), cwd=self.root / "work", env=self.environment(),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.out = queue.Queue()
        threading.Thread(target=_pump, args=(self.proc.stdout, self.out), daemon=True).start()

    def stop(self):
        p, self.proc = self.proc, None
        if p is None:
            return
        kill_tree(p.pid)
        with contextlib.suppress(subprocess.TimeoutExpired):
            p.wait(5)
        with contextlib.suppress(OSError, ValueError):
            p.stdin.close()

    def restart(self):
        self.stop()
        self.start()

    def run(self, command):
        """Run one command in the shell. Returns (output, is_error)."""
        if "\0" in command:
            raise ToolError("the command contains a NUL character")
        if self.proc is None or self.proc.poll() is not None:
            self.stop()
            self.start()
        token = os.urandom(8).hex()
        end = re.compile(rb"\n__GATOR_" + token.encode() + rb"_END__ (\d+)\n")
        # The marker is split by %s, so `set -x` echoing this line cannot fake it.
        script = f"eval -- {shlex.quote(command)} < /dev/null\nprintf '\\n__GATOR_%s_END__ %s\\n' {token} \"$?\"\n"
        try:
            self.proc.stdin.write(script.encode())
            self.proc.stdin.flush()
        except OSError:
            self.restart()
            return "the shell had exited; a new shell was started in /work. Run the command again.", True
        buf, omitted, t_end, drain_end = bytearray(), 0, time.monotonic() + self.timeout, None
        while True:
            now = time.monotonic()
            if drain_end is not None and now >= drain_end:
                chunk = None
            elif drain_end is None and now >= t_end:
                text = cap_output(buf.decode(errors="replace"), omitted)
                self.restart()
                return (text + ("" if not text or text.endswith("\n") else "\n")
                        + f"timed out after {self.timeout:g} s; the shell was restarted"), True
            else:
                try:
                    chunk = self.out.get(timeout=max(0.0, (drain_end or min(now + POLL_S, t_end)) - now))
                except queue.Empty:
                    if drain_end is None and self.proc.poll() is not None:
                        # The shell exited, but a job it started in the background still holds its output
                        # open, so no end of file comes: take what is still on the way, then stop.
                        drain_end = time.monotonic() + DRAIN_S
                    continue
            if chunk is None:  # the shell exited (exit, set -e) or closed its output
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self.proc.wait(5)
                code = self.proc.returncode
                self.stop()
                text = cap_output(buf.decode(errors="replace"), omitted)
                return (text + ("" if not text or text.endswith("\n") else "\n")
                        + f"[the shell ended{f' with code {code}' if code is not None else ''}; the next command "
                          "starts a new shell in /work, without the earlier directory changes and variables]"), False
            n = len(buf)
            buf += chunk
            m = end.search(buf, max(0, n - 64))
            if m:
                return _with_code(cap_output(buf[:m.start()].decode(errors="replace"), omitted), int(m[1])), False
            if len(buf) > TRIM_AT:
                cut = len(buf) - KEEP_TAIL
                omitted += cut - KEEP_HEAD
                del buf[KEEP_HEAD:cut]


def bash_tool(shell, inp):
    if not isinstance(inp, dict):
        raise ToolError('bash takes {"command": "..."} or {"restart": true}')
    if inp.get("restart") is True:
        shell.restart()
        return "the shell was restarted: a new bash in /work, without the earlier directory changes and variables", False
    cmd = inp.get("command")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ToolError('bash takes {"command": "..."} or {"restart": true}')
    return shell.run(cmd)


# ---------------------------------------------------------------------------------------------- files


class WorkFiles:
    """Files under root/work for the text editor and view_image, which run on the host.

    A path is /work/..., the same file at its host path, or relative to /work. It must not contain '..'
    and must resolve inside root/work. Files are then opened component by component from root/work with
    O_NOFOLLOW, so a symlink swapped in after the check cannot lead out of it either.
    """

    def __init__(self, work):
        self.work = Path(work).resolve()

    def resolve(self, raw):
        """(host path, path relative to work) of a path the model gave."""
        if not isinstance(raw, str) or not raw.strip():
            raise ToolError("path must be a non-empty string")
        if "\0" in raw:
            raise ToolError("path contains a NUL character")
        p = PurePosixPath(raw)
        if p.is_absolute():
            for base in ("/work", self.work.as_posix()):
                with contextlib.suppress(ValueError):
                    p = p.relative_to(base)
                    break
            else:
                raise ToolError(f"{raw} is outside /work")
        if ".." in p.parts:
            raise ToolError(f"{raw}: '..' is not allowed; give a path inside /work")
        host = (self.work / p).resolve()
        if not host.is_relative_to(self.work):
            raise ToolError(f"{raw} leads outside /work")
        return host, PurePosixPath(host.relative_to(self.work).as_posix())

    @staticmethod
    def show(rel):
        return "/work" + (f"/{rel}" if rel.parts else "")

    def open(self, rel, flags, make_parents=False):
        """os.open of work/rel without following a symlink anywhere below work."""
        fd = os.open(self.work, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for part in rel.parts[:-1]:
                if make_parents:
                    with contextlib.suppress(FileExistsError):
                        os.mkdir(part, 0o755, dir_fd=fd)
                sub = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = sub
            return os.open(rel.parts[-1], flags | os.O_NOFOLLOW, 0o644, dir_fd=fd) if rel.parts else os.dup(fd)
        finally:
            os.close(fd)

    def read(self, rel, limit=MAX_READ_BYTES):
        fd = self.open(rel, os.O_RDONLY | os.O_NONBLOCK)
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise ToolError(f"{self.show(rel)} is not a regular file")
            if st.st_size > limit:
                raise ToolError(f"{self.show(rel)} has {st.st_size} bytes; at most {limit} can be read here")
            with os.fdopen(os.dup(fd), "rb") as f:
                return f.read(limit + 1)
        finally:
            os.close(fd)

    def write(self, rel, data, make_parents=False):
        fd = self.open(rel, os.O_WRONLY | os.O_CREAT | os.O_NONBLOCK, make_parents)
        with os.fdopen(fd, "wb") as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
                raise ToolError(f"{self.show(rel)} is not a regular file")
            f.truncate(0)
            f.write(data)


def _lines(text):
    """Lines of text without their newlines; a final newline does not start another line."""
    lines = text.split("\n")
    return lines[:-1] if lines[-1] == "" else lines


def _numbered(lines, first, last):
    return "\n".join(f"{i:6}\t{lines[i - 1]}" for i in range(first, last + 1))


def _index(x):
    return isinstance(x, int) and not isinstance(x, bool)


class TextEditor:
    """The str_replace_based_edit_tool commands view, create, str_replace and insert on root/work."""

    def __init__(self, files, max_chars=MAX_VIEW_CHARS):
        self.files, self.max_chars, self.depth = files, max_chars, 2

    def __call__(self, inp):
        if not isinstance(inp, dict):
            raise ToolError("the input must be an object with a command")
        fn = {"view": self.view, "create": self.create, "str_replace": self.str_replace,
              "insert": self.insert}.get(inp.get("command"))
        if fn is None:
            raise ToolError(f"unknown command {inp.get('command')!r}; use view, create, str_replace or insert")
        try:
            return fn(inp)
        except OSError as e:
            raise ToolError(f"{inp.get('path')}: {e.strerror or e}") from None

    def clip(self, text, what="view_range"):
        if len(text) <= self.max_chars:
            return text
        return text[:self.max_chars] + f"\n[cut at {self.max_chars} characters; use {what} to see the rest]"

    def text(self, rel):
        data = self.files.read(rel)
        if b"\0" in data:
            raise ToolError(f"{self.files.show(rel)}: binary file; use view_image for PNG or JPEG")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            raise ToolError(f"{self.files.show(rel)}: binary file; use view_image for PNG or JPEG") from None

    def view(self, inp):
        host, rel = self.files.resolve(inp.get("path"))
        shown = self.files.show(rel)
        try:
            st = os.lstat(host)
        except FileNotFoundError:
            raise ToolError(f"{shown} does not exist") from None
        if stat.S_ISDIR(st.st_mode):
            if inp.get("view_range") is not None:
                raise ToolError(f"{shown} is a folder; view_range applies to files")
            return self.listing(rel)
        lines = _lines(self.text(rel))
        rng = inp.get("view_range")
        first, last = 1, len(lines)
        if rng is not None:
            if not (isinstance(rng, list) and len(rng) == 2 and all(map(_index, rng))):
                raise ToolError("view_range must be [first, last] line numbers (last -1 for the end)")
            first, last = rng[0], len(lines) if rng[1] == -1 else rng[1]
            if not (1 <= first <= len(lines) and first <= last <= len(lines)):
                raise ToolError(f"view_range {rng} is outside {shown}, which has {len(lines)} lines")
        if not lines:
            return f"{shown} is empty"
        return self.clip(_numbered(lines, first, last))

    def listing(self, rel):
        out = []

        def walk(fd, prefix, level):
            for name in sorted(os.listdir(fd)):
                if name.startswith("."):
                    continue
                st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                path = f"{prefix}/{name}"
                if stat.S_ISDIR(st.st_mode):
                    out.append(path + "/")
                    if level < self.depth:
                        try:
                            sub = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                        except OSError as e:
                            out[-1] += f" ({e.strerror})"
                            continue
                        try:
                            walk(sub, path, level + 1)
                        finally:
                            os.close(sub)
                elif stat.S_ISLNK(st.st_mode):
                    out.append(path + "@ (symlink)")
                else:
                    out.append(f"{path} ({st.st_size} bytes)")

        fd = self.files.open(rel, os.O_RDONLY | os.O_DIRECTORY)
        try:
            walk(fd, self.files.show(rel), 1)
        finally:
            os.close(fd)
        head = f"{self.files.show(rel)}/, {self.depth} levels deep, hidden files left out:\n"
        return self.clip(head + ("\n".join(out) or "(empty)"), "a narrower path")

    def create(self, inp):
        text = inp.get("file_text")
        if not isinstance(text, str):
            raise ToolError("create needs file_text, the whole file")
        host, rel = self.files.resolve(inp.get("path"))
        if not rel.parts or host.is_dir():
            raise ToolError(f"{self.files.show(rel)} is a folder")
        existed = host.exists()
        self.files.write(rel, text.encode("utf-8"), make_parents=True)
        return f"{'overwrote' if existed else 'created'} {self.files.show(rel)} ({len(_lines(text))} lines)"

    def str_replace(self, inp):
        old, new = inp.get("old_str"), inp.get("new_str", "")
        if not isinstance(old, str) or not old:
            raise ToolError("str_replace needs old_str, text that occurs exactly once in the file")
        if not isinstance(new, str):
            raise ToolError("new_str must be a string")
        _, rel = self.files.resolve(inp.get("path"))
        shown, text = self.files.show(rel), self.text(rel)
        n = text.count(old)
        if n != 1:
            raise ToolError(f"old_str was not found in {shown}" if n == 0 else
                            f"old_str occurs {n} times in {shown}; include more lines to make it unique")
        i = text.index(old)
        text = text[:i] + new + text[i + len(old):]
        self.files.write(rel, text.encode("utf-8"))
        first = text.count("\n", 0, i) + 1
        return f"edited {shown}:\n" + self.snippet(text, first, first + new.count("\n") - new.endswith("\n"))

    def insert(self, inp):
        k, ins = inp.get("insert_line"), inp.get("insert_text", inp.get("new_str"))
        if not _index(k):
            raise ToolError("insert needs insert_line, the line to insert after (0 for the top)")
        if not isinstance(ins, str):
            raise ToolError("insert needs insert_text")
        _, rel = self.files.resolve(inp.get("path"))
        shown, text = self.files.show(rel), self.text(rel)
        lines = _lines(text)
        if not 0 <= k <= len(lines):
            raise ToolError(f"insert_line {k} is outside 0 to {len(lines)}, the lines of {shown}")
        new = _lines(ins) or [""]
        out = "\n".join(lines[:k] + new + lines[k:])
        out += "\n" if text.endswith("\n") or not text else ""
        self.files.write(rel, out.encode("utf-8"))
        return f"inserted {len(new)} lines into {shown} after line {k}:\n" + self.snippet(out, k + 1, k + len(new))

    def snippet(self, text, first, last):
        lines = _lines(text)
        if not lines:
            return "(the file is now empty)"
        first, last = max(1, first - 3), min(len(lines), max(first, last) + 3)
        return self.clip(_numbered(lines, min(first, last), last))


def _has_alpha(im):
    return im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info


def b64_size(n):
    """Length of the base64 of n bytes."""
    return 4 * ((n + 2) // 3)


def _jpeg(im):
    """im as JPEG at the first of JPEG_QUALITIES whose base64 fits MAX_IMAGE_B64 (else the lowest).
    Transparent areas become white, as on review.py's sheet."""
    from PIL import Image

    if _has_alpha(im):
        im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im.convert("RGBA"))
    im = im.convert("L" if im.mode == "L" else "RGB")
    for q in JPEG_QUALITIES:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=q)
        if b64_size(buf.tell()) <= MAX_IMAGE_B64:
            break
    return buf.getvalue()


class ImageBudget:
    """The images of a session: every one stays in the history and goes with every later request, so
    their base64 bytes and their number are capped (IMAGE_BUDGET_BYTES, MAX_IMAGES)."""

    def __init__(self, limit=None):
        self.limit = IMAGE_BUDGET_BYTES if limit is None else limit
        self.count = MAX_IMAGES
        self.used = self.n = 0

    def take(self, content):
        """Count the image of a view_image result, or ToolError when it does not fit."""
        size = len(content[0]["source"]["data"])
        if self.n >= self.count or self.used + size > self.limit:
            raise ToolError(
                f"not shown: the conversation already holds {self.n} images ({self.used / 1e6:.1f} MB), and every image "
                f"stays in it and goes with each request; a session can show at most {self.count} images and "
                f"{self.limit / 1e6:.0f} MB of them (this one needs {size / 1e6:.2f} MB). Check the result with numbers "
                "from the review files instead, or look at a smaller image.")
        self.used, self.n = self.used + size, self.n + 1


def view_image(files, inp):
    """The view_image tool, downscaling to at most MAX_IMAGE_SIDE px on the long side. Returns
    (tool_result content, (W, H), shown (w, h)).

    The input is validated here, because eager input streaming skips the server's validation. A file
    goes as it is when it needs no downscaling and its base64 fits MAX_IMAGE_B64; everything else is
    encoded again, a PNG as PNG if that fits MAX_IMAGE_B64, else as JPEG (_jpeg).
    """
    if not (isinstance(inp, dict) and set(inp) == {"path"} and isinstance(inp["path"], str)):
        raise ToolError('view_image takes {"path": "<a PNG or JPEG under /work>"}')
    from PIL import Image

    _, rel = files.resolve(inp["path"])
    shown = files.show(rel)
    try:
        data = files.read(rel)
    except FileNotFoundError:
        raise ToolError(f"{shown} does not exist") from None
    if data.startswith(PNG_MAGIC):
        kind = "PNG"
    elif data.startswith(JPEG_MAGIC):
        kind = "JPEG"
    else:
        raise ToolError(f"{shown} is not a PNG or JPEG file; view_image shows only those")
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:  # PIL raises many kinds for broken files
        raise ToolError(f"{shown} could not be read as an image: {e}") from None
    W, H = im.size
    k = min(1.0, MAX_IMAGE_SIDE / max(W, H))
    size = (max(1, round(W * k)), max(1, round(H * k)))
    note = ""
    if k < 1.0 or b64_size(len(data)) > MAX_IMAGE_B64:
        if im.mode not in ("RGB", "RGBA", "L", "LA"):
            im = im.convert("RGBA" if _has_alpha(im) else "RGB")
        if k < 1.0:
            im = im.resize(size, Image.LANCZOS)
        buf = io.BytesIO()
        if kind == "PNG":
            im.save(buf, "PNG")
        if kind == "PNG" and b64_size(buf.tell()) <= MAX_IMAGE_B64:
            data = buf.getvalue()
        else:
            if kind == "PNG":
                note = ", sent as JPEG" + (" with transparent areas white" if _has_alpha(im) else "")
            kind, data = "JPEG", _jpeg(im)
    if len(data) > API_IMAGE_BYTES:
        raise ToolError(f"{shown} is too large to send, also as JPEG ({len(data)} bytes)")
    media = "image/png" if kind == "PNG" else "image/jpeg"
    content = [{"type": "image", "source": {"type": "base64", "media_type": media,
                                            "data": base64.standard_b64encode(data).decode("ascii")}},
               {"type": "text", "text": f"{shown} {W}x{H}, shown at {size[0]}x{size[1]}{note}"}]
    return content, (W, H), size


# ----------------------------------------------------------------------------------------------- loop


def request_params(model, effort, system):
    """Everything of a request but messages. Built once per session, so tools and system stay
    byte-stable and the cache and the thinking blocks stay valid."""
    return {"model": model, "max_tokens": MAX_TOKENS,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "tools": TOOLS, "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": effort}, "cache_control": {"type": "ephemeral"}, "betas": list(BETAS),
            "fallbacks": "default"}


def stream_turn(client, params, log, turn):
    """One streamed request. Returns (message, seconds of the attempt that succeeded, number of
    attempts that failed before it).

    Nothing of a failed attempt goes into the history, so the same request is sent again (the history
    stays append-only) when:
    - the SDK raised ValueError: it could not parse a tool input that eager streaming sent, and there
      is no tool_use_id to answer; at most JSON_RETRIES times;
    - the stream broke off after it opened, which the SDK's own retries (of the request) do not cover:
      a retryable error event (an APIStatusError with the stream's status, 200), a dropped connection
      (a raw httpx2 transport error) or a stream that ended without a stop reason; at most
      STREAM_RETRIES times, after RETRY_BACKOFF_S, doubled each time. Then StreamFailed.
    Errors before the stream opened propagate: the SDK has already retried the retryable ones.
    """
    json_fails = stream_fails = 0
    for attempt in itertools.count(1):
        log.write("request", turn=turn, **({"attempt": attempt} if attempt > 1 else {}))
        t, opened = time.time(), False
        try:
            with client.beta.messages.stream(**params) as stream:
                opened = True
                msg = stream.get_final_message()
            if msg.stop_reason is not None:
                return msg, time.time() - t, attempt - 1
            error = "the stream ended without a stop reason, so the response is incomplete"
        except ValueError as e:
            json_fails += 1
            log.write("retry", turn=turn, attempt=attempt, error=str(e)[:1000])
            if json_fails > JSON_RETRIES:
                raise BadToolJSON(str(e)[:1000], attempt) from e
            continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError, httpx2.TransportError) as e:
            if not opened or (isinstance(e, anthropic.APIStatusError) and e.type not in RETRY_TYPES):
                raise
            kind = f" ({e.type})" if isinstance(e, anthropic.APIStatusError) else ""
            error = f"the stream broke off: {type(e).__name__}{kind}: {e}"
        stream_fails += 1
        log.write("retry", turn=turn, attempt=attempt, error=error[:1000])
        if stream_fails > STREAM_RETRIES:
            raise StreamFailed(error[:1000], attempt)
        time.sleep(RETRY_BACKOFF_S * 2 ** (stream_fails - 1))


def _tokens(u):
    return {k: getattr(u, k, None) or 0 for k in tools.TOKENS}


def _attempt_cost(u, model):
    """USD of one attempt's usage u at the prices of model (the dearest listed one if not listed). A
    cache write is a 5-minute one unless u.cache_creation says it was a 1-hour one."""
    p = tools.PRICES.get(model) or max(tools.PRICES.values(), key=lambda x: x["output_tokens"])
    t = _tokens(u)
    hour = getattr(getattr(u, "cache_creation", None), "ephemeral_1h_input_tokens", 0) or 0
    long = min(t["cache_creation_input_tokens"], hour)
    return (sum(t[k] * p[k] for k in tools.TOKENS) + long * (p["cache_creation_1h"] - p["cache_creation_input_tokens"])) / 1e6


def billed(msg, model):
    """(tokens, cost_usd, iterations) of one response. The top-level usage covers only the attempt that
    produced the message; with a refusal fallback usage.iterations lists every attempt (the declined
    one bills at normal rates, the fallback at its model's). Each attempt is priced by its model, else
    the model asked for; without iterations the usage is priced by msg.model. iterations: the attempts
    for the log, [] without them."""
    its = getattr(msg.usage, "iterations", None) or []
    parts = [(e, getattr(e, "model", None) or model) for e in its] or [(msg.usage, msg.model or model)]
    tokens = {k: sum(_tokens(u)[k] for u, _ in parts) for k in tools.TOKENS}
    cost = sum(_attempt_cost(u, m) for u, m in parts)
    return tokens, cost, [{"type": e.type, "model": m, **_tokens(e)} for e, m in parts if its]


def echo_content(content):
    """The assistant turn to send back: content unchanged, unless a fallback model took over mid-output.
    Then only text and fallback blocks come before the last fallback block (the API's echo rule: the
    declined model's thinking and its tool calls, which never ran, are left out)."""
    last = max((i for i, b in enumerate(content) if b.type == "fallback"), default=None)
    if last is None:
        return content
    kept = [b for i, b in enumerate(content) if i >= last or b.type in ("text", "fallback")]
    return content if len(kept) == len(content) else kept


def log_blocks(content):
    out = []
    for b in content:
        if b.type == "thinking":
            out.append({"type": "thinking", "text": b.thinking})
        elif b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        elif b.type == "fallback":
            out.append(b.to_dict())
        else:
            out.append({"type": b.type})
    return out


def _short(text, n=200):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "..."


def run_tool(block, shell, editor, files, images=None):
    """Run one tool_use block. Returns (content, is_error, log text). Never raises a tool's error: every
    tool_use gets its result. images: the session's ImageBudget."""
    try:
        if block.name == "bash":
            text, is_error = bash_tool(shell, block.input)
        elif block.name == "str_replace_based_edit_tool":
            text, is_error = editor(block.input), False
        elif block.name == "view_image":
            content, (W, H), (w, h) = view_image(files, block.input)
            (images or ImageBudget()).take(content)
            return content, False, f"[image {W}x{H}]"
        else:
            raise ToolError(f"unknown tool {block.name!r}")
    except ToolError as e:
        text, is_error = str(e), True
    except Exception as e:  # a bug or an OS error must not drop the result
        text, is_error = f"{type(e).__name__}: {e}", True
    return text, is_error, text


def _interrupt(signum, frame):
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, signal.SIG_IGN)
    raise Interrupted(signum)


def run(root, sif=None, sock_dir=None, key_file=None, *, effort="high", model=MODEL, shell="container",
        client=None, command_timeout=COMMAND_TIMEOUT_S, out=None):
    """Run the session in root (prepared by controller.prepare_session). Returns the exit code.

    client: an anthropic.Anthropic or a stand-in with beta.messages.stream; by default one is made with
    the key from key_file. The key from key_file, when given, is also redacted from the log, stdout and
    the tool results.
    """
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}")
    root, out = Path(root).resolve(), out or sys.stdout
    key = read_key(key_file) if key_file else None
    log = Log(root / "session" / "claude.jsonl", key)
    sh = ContainerShell(root, sif, sock_dir, host=shell == "host", timeout=command_timeout)
    files = WorkFiles(root / "work")
    editor, images = TextEditor(files), ImageBudget()
    totals, responses, cost, reissued = dict.fromkeys(tools.TOKENS, 0), 0, 0.0, 0

    def say(text):
        print(redact(text, key), file=out, flush=True)

    def end(reason, code, **detail):
        # reissued: attempts that failed and were sent again; they are billed too, but their usage is unknown.
        log.write("end", reason=reason, exit_code=code, turns=responses, totals=totals, cost_usd=round(cost, 4),
                  reissued=reissued, **detail)
        say(f"end: {reason} (exit {code}), {responses} turns, ${cost:.2f} estimated"
            + (f" (at least; {reissued} failed attempts not counted)" if reissued else "")
            + (f": {detail['message']}" if detail.get("message") else ""))
        return code

    main_thread = threading.current_thread() is threading.main_thread()
    old = {s: signal.signal(s, _interrupt) for s in (signal.SIGINT, signal.SIGTERM)} if main_thread else {}
    try:
        try:
            if client is None:
                if not key:
                    return end("no_key", EXIT["api_error"], message=f"no ANTHROPIC_API_KEY= line in {key_file}")
                client = anthropic.Anthropic(api_key=key, max_retries=MAX_RETRIES)
            system_text = SYSTEM_FILE.read_text(encoding="utf-8")
            params = request_params(model, effort, system_text)
            messages = [{"role": "user", "content": (root / "session" / "prompt.txt").read_text(encoding="utf-8")}]
            log.write("start", model=model, effort=effort, betas=params["betas"], shell=shell,
                      max_tokens=MAX_TOKENS, image_side=MAX_IMAGE_SIDE, system=str(SYSTEM_FILE),
                      system_sha256=hashlib.sha256(system_text.encode("utf-8")).hexdigest())
            turn = pauses = 0
            while True:
                turn += 1
                msg, seconds, failed = stream_turn(client, {**params, "messages": list(messages)}, log, turn)
                responses, reissued = responses + 1, reissued + failed
                usage = _tokens(msg.usage)
                tokens, turn_cost, iterations = billed(msg, model)
                cost += turn_cost
                for k in tools.TOKENS:
                    totals[k] += tokens[k]
                stop_details = msg.stop_details.to_dict() if msg.stop_details is not None else None
                log.write("response", turn=turn, id=msg.id, model=msg.model, stop_reason=msg.stop_reason,
                          stop_details=stop_details, usage=usage, seconds=round(seconds, 3),
                          blocks=log_blocks(msg.content), cost_usd=round(turn_cost, 6),
                          **({"iterations": iterations, "billed": tokens} if iterations else {}))
                say(f"turn {turn}: {msg.stop_reason}, {seconds:.1f} s, {msg.model}, in {usage['input_tokens']} "
                    f"out {usage['output_tokens']} cache read {usage['cache_read_input_tokens']} "
                    f"write {usage['cache_creation_input_tokens']}"
                    + "".join(f"; {e['type']} {e['model']} out {e['output_tokens']}" for e in iterations))
                for b in log_blocks(msg.content):
                    say(f"  {b['type']}: " + _short(b.get("text", b.get("input", b)), 300))
                if msg.stop_reason == "refusal":
                    # A refusal can cut a tool_use off mid-input: nothing of this turn runs.
                    return end("refusal", EXIT["refusal"], stop_details=stop_details)
                content = echo_content(msg.content)
                if msg.stop_reason == "pause_turn":
                    pauses += 1
                    if pauses > MAX_PAUSE_TURNS:
                        return end("pause_turn_limit", EXIT["api_error"],
                                   message=f"the API paused the turn {pauses} times in a row")
                    messages.append({"role": "assistant", "content": content})
                    continue
                pauses = 0
                if msg.stop_reason == "model_context_window_exceeded":
                    # The history is append-only, so it cannot be made shorter: the session ends here, and
                    # the turn's tool calls, which may be cut off, do not run.
                    return end(msg.stop_reason, EXIT["ended"], message="the conversation filled the context window")
                uses = [b for b in content if b.type == "tool_use"]
                if not uses:
                    return end(msg.stop_reason, EXIT["ended"])
                messages.append({"role": "assistant", "content": content})
                results = []
                for b in uses:
                    t = time.time()
                    if msg.stop_reason != "tool_use":
                        # Only a turn that stopped for its tool calls has complete inputs. The tolerant parser
                        # turns a cut-off input into a valid-looking object (a str_replace without new_str
                        # would delete old_str): never run it.
                        res = (f"not run: this call was cut off at the output limit ({MAX_TOKENS} tokens), so its "
                               "input is incomplete. Make it again, in smaller pieces if it was long."
                               if msg.stop_reason == "max_tokens" else
                               f"not run: the turn stopped with stop reason {msg.stop_reason!r}, so this call's input "
                               "may be incomplete. Make it again.")
                        is_error, logged = True, res
                    else:
                        res, is_error, logged = run_tool(b, sh, editor, files, images)
                    if isinstance(res, str):
                        res = redact(res, key)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": res,
                                    **({"is_error": True} if is_error else {})})
                    log.write("tool_result", turn=turn, tool_use_id=b.id, name=b.name,
                              seconds=round(time.time() - t, 3), is_error=is_error, output=logged[:LOG_OUTPUT_CHARS])
                    say(f"  -> {b.name}{' error' if is_error else ''}: {_short(logged)}")
                messages.append({"role": "user", "content": results})
                if any(e.get("event") == "finish" for e in tools.read_events(root / "control")):
                    return end("finished", EXIT["ended"])
        except anthropic.APIError as e:
            label = next(text for cls, text in API_ERRORS if isinstance(e, cls))
            return end("api_error", EXIT["api_error"], message=label, error={
                "class": type(e).__name__, "status": getattr(e, "status_code", None), "type": getattr(e, "type", None),
                "request_id": getattr(e, "request_id", None), "detail": str(e)[:1000]})
        except BadToolJSON as e:
            reissued += e.failed
            return end("invalid_tool_json", EXIT["api_error"],
                       message=f"the tool input could not be parsed after {JSON_RETRIES} re-issues", error=str(e))
        except StreamFailed as e:
            reissued += e.failed
            return end("api_error", EXIT["api_error"], error=str(e),
                       message=f"the response stream broke off, also after {STREAM_RETRIES} re-issues")
        except Exception as e:  # a bug or a missing session file: log it, then fail with the traceback
            end("error", 1, message=f"{type(e).__name__}: {e}"[:1000])
            raise
    except Interrupted as e:
        return end("interrupted", EXIT["interrupted"], signal=signal.Signals(e.args[0]).name)
    finally:
        if main_thread:  # no second interrupt while the shell is stopped
            for s in old:
                signal.signal(s, signal.SIG_IGN)
        sh.stop()
        log.close()
        for s, h in old.items():
            signal.signal(s, signal.SIG_DFL if h is None else h)


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the session folder from controller.prepare_session")
    ap.add_argument("--sif", help="the agent's Apptainer image (not used with --shell host)")
    ap.add_argument("--sock-dir", required=True, help="the folder of the tool server's socket")
    ap.add_argument("--key-file", required=True, help="a file with the line ANTHROPIC_API_KEY=...")
    ap.add_argument("--effort", choices=EFFORTS, default="high")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--shell", choices=("container", "host"), default="container",
                    help="host: bash on the host in ROOT/work, for tests and local runs only")
    a = ap.parse_args(argv)
    if a.shell == "container" and not a.sif:
        ap.error("--sif is required unless --shell host")
    return run(a.root, a.sif, a.sock_dir, a.key_file, effort=a.effort, model=a.model, shell=a.shell, client=client)


if __name__ == "__main__":
    sys.exit(main())
