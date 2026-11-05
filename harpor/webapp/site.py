"""Build a standalone static demo gallery; no accounts or cloud services required."""
import argparse
import base64
import hashlib
import html
import os
import re
import shutil
import sys
from pathlib import Path
from harpor.webapp import demos

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
OUT = HERE.parents[1] / "site"
MODEL_VIEWER = ('<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.4.0/'
                'model-viewer.min.js"></script>')
VENDOR = {
    "licenses/three-0.183.0.txt": "dZIEXnc8kDiJUEWlKdsN+V7yvZT/HbLEL7IFgYWNSRkA0ErUhU5jIL71XpYCHvDY",
    "model-viewer-4.4.0.min.js": "wjsPzEO+lXOf7d9csIDfAQpPt2OKuUU3m0K9qsdKIuzX4N8czSF6hdh23eAbZa3p",
    "licenses/model-viewer-4.4.0.txt": "II9e1ieUDl5AxyiVq3/FflTua1Sr0kMJ25e6imG7rXg7SiAsA2VemsvEqVsLqM7/",
}
# The CSP hash of the one <style> block in model-viewer-4.4.0.min.js: the styles of its shadow DOM, which a policy
# without unsafe-inline blocks otherwise (the browser names this hash). The file is pinned above, so the text is too;
# retain this hash when updating the bundled viewer.
MODEL_VIEWER_STYLE = "'sha256-PXtZZkJMMyc18o/DN1ZEFOJ2qEGq95IQbVaJFMOXOb0='"

class SiteError(Exception):
    """Why the site cannot be built or configured, in one line."""

def b64sha384(data):
    return base64.b64encode(hashlib.sha384(data).digest()).decode()

def csp(meta=False):
    rules = [("default-src", "'none'"), ("script-src", "'self' 'wasm-unsafe-eval'"),
             ("style-src", f"'self' {MODEL_VIEWER_STYLE}"), ("img-src", "'self' data: blob:"),
             ("media-src", "'self' blob:"), ("connect-src", "'self' blob:"),
             ("worker-src", "'self' blob:"), ("font-src", "'self'"), ("manifest-src", "'self'"),
             ("frame-ancestors", "'none'"), ("base-uri", "'none'"), ("form-action", "'none'")]
    return "; ".join(f"{k} {v}" for k, v in rules if not (meta and k == "frame-ancestors"))

def head_meta():
    return (f'<meta http-equiv="Content-Security-Policy" content="{html.escape(csp(meta=True), quote=False)}">\n'
            '<meta name="referrer" content="no-referrer">')

def once(text, old, new, page="static/index.html"):
    """text with old (which must occur exactly once) replaced by new; SiteError names the anchor otherwise, so a
    change in the page stops the build instead of leaving a part out."""
    n = text.count(old)
    if n != 1:
        raise SiteError(f"{page} has {n} copies of {old.strip()[:80]!r}, not 1; update site.py")
    return text.replace(old, new)

def public_demos(src, shown, soon=()):
    """static/demos.html made into the public site's demos.html, with the demos shown and the placeholders' cards."""
    page = "static/demos.html"
    out = once(src, '<meta charset="utf-8">', '<meta charset="utf-8">\n' + head_meta(), page)
    out = once(out, 'href="/static/style.css"', 'href="static/style.css"', page)
    out = once(out, 'href="/"', 'href="index.html"', page)  # the brand link
    out = once(out, MODEL_VIEWER, '<script type="module" src="static/vendor/model-viewer-4.4.0.min.js"></script>', page)
    out = once(out, 'src="/static/demos.js"', 'src="static/demos.js"', page)
    try:
        out = demos.page(out, shown, soon)
    except demos.DemoError as e:
        raise SiteError(str(e)) from None
    if re.search(r'(?:src|href)="(?:/|https?:)', out):
        raise SiteError("the public demos.html still has an absolute path or another host; update site.py")
    return out

def headers_file():
    return ("# Response headers for Cloudflare Pages, written by harpor/webapp/site.py. GitHub Pages ignores this\n"
            "# file; the pages carry the same policy in a <meta> tag, without frame-ancestors, which a <meta> cannot\n"
            "# set.\n"
            "/*\n"
            f"  Content-Security-Policy: {csp()}\n"
            "  Referrer-Policy: no-referrer\n"
            "  Strict-Transport-Security: max-age=31536000\n"
            "  X-Content-Type-Options: nosniff\n"
            "  Permissions-Policy: camera=(), microphone=(), geolocation=()\n"
            "\n"
            "# The vendor files have their version in their name, so they never change.\n"
            "/static/vendor/*\n"
            "  Cache-Control: public, max-age=31536000, immutable\n")

def vendor_files(static=STATIC):
    """{path under static/vendor: bytes}, each checked against VENDOR."""
    out = {}
    for rel, want in VENDOR.items():
        p = Path(static) / "vendor" / rel
        if not p.is_file():
            raise SiteError(f"static/vendor/{rel} is missing (static/vendor/README.md says where it comes from)")
        data = p.read_bytes()
        if b64sha384(data) != want:
            raise SiteError(f"static/vendor/{rel} does not have its SHA-384; take it again from its npm release")
        out[rel] = data
    return out

def files(static=STATIC, demo_dir=None):
    try:
        shown = demos.load(demo_dir)
        soon = demos.placeholders(demo_dir, shown)
    except demos.DemoError as e:
        raise SiteError(str(e)) from e
    page = public_demos((Path(static) / "demos.html").read_text(), shown, soon).encode()
    help_page = (Path(static) / "help.html").read_text().replace('href="/"', 'href="index.html"').replace('href="/static/', 'href="static/')
    help_page = help_page.replace('<meta charset="utf-8">', '<meta charset="utf-8">\n' + head_meta())
    out = {"index.html": page, "demos.html": page, "help.html": help_page.encode(),
           "_headers": headers_file().encode(), ".nojekyll": b"", ".harpor-gallery": b""}
    out.update({rel: p.read_bytes() for rel, p in demos.files(demo_dir, shown).items()})
    for name in ("style.css", "demos.js"):
        out[f"static/{name}"] = (Path(static) / name).read_bytes()
    out.update({f"static/vendor/{rel}": data for rel, data in vendor_files(static).items()})
    return out

def build(out=OUT, static=STATIC, demo_dir=None, no_demos=False):
    """Write the site into out, with the demos of demo_dir. An existing out is replaced only when it is an earlier
    build (or empty), and one with demos or placeholders' cards only when demo_dir is given or no_demos is true."""
    out = Path(out)
    site = files(static, demo_dir)
    if out.exists():
        if any(out.iterdir()) and not ((out / ".nojekyll").is_file() and ((out / ".harpor-gallery").is_file() or (out / "static" / "config.js").is_file())):
            raise SiteError(f"{out} is not an earlier build of the site; choose another --out")
        page = out / "demos.html"
        cards = page.is_file() and 'class="democard' in page.read_text(errors="replace")
        if demo_dir is None and not no_demos and ((out / "demos").is_dir() or cards):
            raise SiteError(f"the earlier build in {out} has demos or placeholders; give --demos DIR, or --no-demos to "
                            "build without them")
        shutil.rmtree(out)
    for path, data in site.items():
        p = out / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return sorted(site)

def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m harpor.webapp.site", description="Build the public HARPOR site.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the site folder")
    b.add_argument("--out", default=str(OUT), help=f"the folder to write (default {OUT})")
    d = b.add_mutually_exclusive_group()
    d.add_argument("--demos", metavar="DIR", default=os.environ.get("WEBAPP_DEMOS"),
                   help="the demos folder (default: WEBAPP_DEMOS)")
    d.add_argument("--no-demos", action="store_true", help="build without demos, also over a build that had them")
    args = ap.parse_args(argv)
    try:
        folder = Path(args.demos).expanduser() if args.demos and not args.no_demos else None
        written = build(args.out, demo_dir=folder, no_demos=args.no_demos)
        print(f"wrote {len(written)} files to {args.out}")
        # A build without the folder publishes an empty gallery, so the line says where the demos came from.
        n = len({p.split("/")[1] for p in written if p.startswith("demos/")})
        soon = (Path(args.out) / "demos.html").read_text().count('class="democard soon"')
        more = f", {soon} placeholder{'s' if soon != 1 else ''}" if soon else ""
        print(f"demos: {n} from {folder}{more}" if folder else "demos: none" if args.no_demos
              else "demos: none (give --demos DIR or set WEBAPP_DEMOS)")
    except SiteError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
