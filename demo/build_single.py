"""Pack the demo page, its bundle and every face crop into one self-contained HTML file.

  python demo/build_single.py            # writes demo/WhoCat_demo.html (needs demo/web/bundle)

The file opens by double-click and shows the sample homes. Taking photos of your own cat needs
demo/server.py, because the face models run there. The file contains PetFace crops: research use only.
"""
import base64
import json
import mimetypes
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent / "web"


def data_url(p):
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def main(out=Path(__file__).resolve().parent / "WhoCat_demo.html"):
    bdir = WEB / "bundle"
    if not (bdir / "bundle.json").exists():
        sys.exit(f"No bundle in {bdir}.")
    bundle = json.loads((bdir / "bundle.json").read_text())
    images = {p.relative_to(bdir).as_posix(): data_url(p)
              for sub in ("img", "img_door") if (bdir / sub).is_dir() for p in sorted((bdir / sub).iterdir())}
    html = (WEB / "index.html").read_text(encoding="utf-8")
    css = (WEB / "app.css").read_text(encoding="utf-8")
    js = (WEB / "app.js").read_text(encoding="utf-8")
    html = html.replace('<link rel="stylesheet" href="app.css">', f"<style>\n{css}\n</style>")
    html = html.replace('src="method.png"', f'src="{data_url(WEB / "method.png")}"')
    payload = ("window.WHOCAT_BUNDLE = " + json.dumps(bundle, separators=(",", ":")) + ";\n"
               + "window.WHOCAT_IMAGES = " + json.dumps(images, separators=(",", ":")) + ";\n")
    html = html.replace('<script src="app.js"></script>', f"<script>\n{payload}</script>\n<script>\n{js}\n</script>")
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(images)} images)")


if __name__ == "__main__":
    main()
