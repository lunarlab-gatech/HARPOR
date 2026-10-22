"""Logging proxy for the agent CLI pilot checks (build_plan.md note 5).

The Codex CLI is pointed at it through a custom provider (controller.codex_config, see note 5). Each
request is forwarded unchanged to --upstream, and the response is streamed back. One JSON line per
request goes to --log with what the pilots need, and nothing secret (no headers, no query string):
  reasoning, model  OpenAI Responses request's reasoning and model  (is the effort xhigh?)
  image_parts       image parts in the request                      (does the PNG reach the model?)
  ttfb_s, total_s   time to the first response byte and in all      (long turns, queueing)
  body_sha1         a repeated hash means the CLI replayed a request

    python3 logproxy.py --port 18080 --log requests.jsonl [--upstream URL]
"""

import argparse
import hashlib
import http.server
import json
import socket
import threading
import time
import urllib.error
import urllib.request

UPSTREAM = "https://api.openai.com"  # the default; --upstream changes it
DROP = {"host", "content-length", "accept-encoding", "connection"}


def _openai_images(node):
    """input_image items anywhere in an OpenAI Responses input."""
    if isinstance(node, dict):
        return (node.get("type") in ("input_image", "image_url")) + sum(_openai_images(v) for v in node.values())
    if isinstance(node, list):
        return sum(_openai_images(v) for v in node)
    return 0


def summarize(body):
    try:
        req = json.loads(body)
    except ValueError:
        return {}
    if not isinstance(req, dict) or ("input" not in req and "reasoning" not in req):  # not a Responses request
        return {}
    inp = req.get("input")
    return {"model": req.get("model"), "reasoning": req.get("reasoning"), "image_parts": _openai_images(inp),
            "turns": len(inp) if isinstance(inp, list) else 1}


def make_handler(log_path, upstream=UPSTREAM):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # close-delimited responses, so streaming needs no chunking

        def _forward(self):
            t0 = time.time()
            n = int(self.headers.get("content-length") or 0)
            body = self.rfile.read(n) if n else None
            headers = {k: v for k, v in self.headers.items() if k.lower() not in DROP}
            req = urllib.request.Request(upstream + self.path, data=body, headers=headers, method=self.command)
            entry = {"t": round(t0, 3), "method": self.command, "path": self.path.split("?")[0]}
            if body:
                entry.update(summarize(body), body_sha1=hashlib.sha1(body).hexdigest(), body_bytes=n)
            try:
                resp = urllib.request.urlopen(req, timeout=900)
            except urllib.error.HTTPError as e:
                resp = e
            except (urllib.error.URLError, OSError) as e:  # upstream unreachable: tell the client, keep logging
                entry.update(status=502, error=str(e)[:200], total_s=round(time.time() - t0, 2))
                self.send_error(502, "upstream unreachable")
                with open(log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                return
            entry["status"] = resp.status if hasattr(resp, "status") else resp.code
            self.send_response(entry["status"])
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                    self.send_header(k, v)
            self.end_headers()
            first = True
            try:
                while True:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                    if not chunk:
                        break
                    if first:
                        entry["ttfb_s"] = round(time.time() - t0, 2)
                        first = False
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                entry["client_closed"] = True
            entry["total_s"] = round(time.time() - t0, 2)
            with open(log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _forward

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--log", required=True)
    ap.add_argument("--upstream", default=UPSTREAM)
    a = ap.parse_args()
    handler = make_handler(a.log, a.upstream.rstrip("/"))

    class V6(http.server.ThreadingHTTPServer):
        address_family = socket.AF_INET6

    # Serve both loopbacks, because Node may resolve "localhost" to ::1 first.
    servers = [http.server.ThreadingHTTPServer(("127.0.0.1", a.port), handler)]
    try:
        servers.append(V6(("::1", a.port), handler))
    except OSError:
        pass
    for s in servers[1:]:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    servers[0].serve_forever()


if __name__ == "__main__":
    main()
