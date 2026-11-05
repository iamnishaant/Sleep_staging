"""Pack the report for Overleaf: tex, bib and only the images the tex actually includes.

    python "S7 FINAL REPORT/pack_overleaf.py"   -> S7 FINAL REPORT/S7_FINAL_REPORT_overleaf.zip
"""
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
tex = [HERE / "mythesis.tex", *sorted((HERE / "chapters").glob("*.tex"))]
used = sorted({m for f in tex for m in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}",
                                                    f.read_text(encoding="utf-8"))})
missing = [u for u in used if not (HERE / "images" / u).exists()]
assert not missing, f"missing images: {missing}"
out = HERE / "S7_FINAL_REPORT_overleaf.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in tex:
        z.write(f, f.relative_to(HERE).as_posix())
    z.write(HERE / "mybib.bib", "mybib.bib")
    for u in used:
        z.write(HERE / "images" / u, f"images/{u}")
print(f"{out.name}: {len(tex)} tex, {len(used)} images, {out.stat().st_size / 1e6:.1f} MB")
