"""V-HTML: provenance of every number on a delivered page (ACCEPTANCE.md §5)."""
import json
from pathlib import Path

from agentic.report import vhtml
from agentic.report.fallback import render_fallback

from .conftest import DEV_PACKETS, REPO

HTML = REPO / "AGENTIC AND SIMULATION" / "results" / "golden" / "html"


def _packet(rec):
    return json.loads((DEV_PACKETS / f"{rec}.json").read_text(encoding="utf-8"))


def test_all_golden_report_pages_pass():
    pages = sorted(HTML.glob("*.html"))
    assert len(pages) == 30
    for p in pages:
        assert vhtml.check(p.read_text(encoding="utf-8"), _packet(p.stem)) == [], p.stem


def test_fallback_pages_pass_for_every_dev_night():
    for f in sorted(DEV_PACKETS.glob("*.json")):
        pk = json.loads(f.read_text(encoding="utf-8"))
        assert vhtml.check(render_fallback(pk, ["F03"]), pk) == [], f.stem


def test_a_changed_number_is_caught():
    page = (HTML / "SC4081E0-PSG.html").read_text(encoding="utf-8")
    pk = _packet("SC4081E0-PSG")
    tampered = page.replace("410 minutes", "412 minutes", 1)
    assert tampered != page
    assert "412" in vhtml.check(tampered, pk)


def test_numbers_inside_codes_are_not_counted():
    assert vhtml.page_numbers("<p>N1 SC4081E0-PSG stage N3 at 7.5</p>") == ["7.5"]


def test_per_epoch_arrays_are_not_sources():
    pk = {"probabilities": {"calibrated": [[0.12345]]}, "evidence_items": []}
    assert vhtml.check("<p>0.1235</p>", pk) == ["0.1235"]
