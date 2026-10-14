"""Drive the running Streamlit app with Playwright, exercise the main features and save
screenshots to docs/screenshots/.  Doubles as an end-to-end smoke test of the UI.

Prerequisites:
    pip install -r requirements-dev.txt
    playwright install chromium        # or use an installed Chrome via --channel chrome
    streamlit run app.py               # in another terminal

Usage:
    python scripts/take_screenshots.py [--url http://localhost:8501] [--channel chrome]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pymupdf
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"


def wait_idle(page: Page, timeout_ms: int = 90_000) -> None:
    """Wait until Streamlit's 'running' status widget disappears and stays gone briefly."""
    deadline = time.time() + timeout_ms / 1000
    quiet_since = None
    while time.time() < deadline:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        if running:
            quiet_since = None
        elif quiet_since is None:
            quiet_since = time.time()
        elif time.time() - quiet_since > 0.8:
            return
        time.sleep(0.2)
    raise TimeoutError("Streamlit did not become idle")


def ask(page: Page, question: str, expect_text: str) -> None:
    box = page.get_by_test_id("stChatInput").locator("textarea")
    box.fill(question)
    box.press("Enter")
    page.get_by_text(expect_text, exact=False).last.wait_for(timeout=90_000)
    wait_idle(page)


def shot(page: Page, name: str, full_page: bool = True, height: int = 900) -> None:
    """Streamlit scrolls inside <section> elements, so a taller viewport (not full_page)
    is what captures a whole conversation."""
    path = OUT / name
    page.set_viewport_size({"width": 1440, "height": height})
    page.evaluate("document.querySelectorAll('section').forEach(s => s.scrollTo(0, 0))")
    time.sleep(0.4)
    page.screenshot(path=str(path), full_page=full_page)
    page.set_viewport_size({"width": 1440, "height": 900})
    print(f"saved {path.relative_to(ROOT)}")


def make_bad_uploads(tmp: Path) -> list[Path]:
    """A scanned (image-only) PDF and a plain-text file renamed .pdf."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), False)
    pix.clear_with(230)
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(72, 72, 540, 500), stream=pix.tobytes("png"))
    scanned = tmp / "scanned_contract.pdf"
    doc.save(scanned)
    doc.close()
    notes = tmp / "meeting_notes.txt"
    notes.write_text("These are plain text notes, not a PDF.")
    return [scanned, notes]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--channel", default="chrome", help="chrome | chromium | msedge (Playwright channel)")
    parser.add_argument("--tmp", type=Path, default=OUT / "_tmp")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.tmp.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel=args.channel, headless=True)
        except Exception:
            browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("AI-Powered Document Assistant").first.wait_for(timeout=120_000)
        wait_idle(page)
        shot(page, "01_home.png", full_page=False)

        # 1) sample-document mode
        page.get_by_role("button", name="Try it with the sample documents").click()
        page.get_by_text("orion_solar_handbook.pdf").first.wait_for(timeout=120_000)
        wait_idle(page)
        shot(page, "02_documents_loaded.png", full_page=False)

        # 2) grounded answer with citations
        ask(page, "What is the community solar credit rate for 2024?", "Sources (")
        assert page.get_by_text("11.7 cents").count() > 0, "expected the credit rate in the answer"
        shot(page, "03_answer_with_sources.png", height=1400)

        # 3) follow-up question
        ask(page, "When are they reconciled each year?", "Follow-up detected")
        assert page.get_by_text("March").count() > 0, "expected the true-up month in the follow-up answer"
        shot(page, "04_followup_question.png", height=1500)

        # 4) question with no evidence
        ask(page, "What is the capital of France?", "couldn't find relevant information")
        shot(page, "05_no_evidence.png", height=1200)

        # 5) scanned PDF + unsupported file messages
        files = make_bad_uploads(args.tmp)
        page.locator('input[type="file"]').set_input_files([str(f) for f in files])
        page.get_by_text("looks like a scanned PDF").first.wait_for(timeout=60_000)
        page.get_by_text("not a PDF file").first.wait_for(timeout=60_000)
        wait_idle(page)
        shot(page, "06_upload_messages.png", full_page=False, height=1300)

        # 6) clear chat
        page.get_by_role("button", name="Clear chat").click()
        wait_idle(page)
        assert page.get_by_test_id("stChatMessage").count() == 0, "chat should be empty after clearing"
        print("UI smoke test passed")
        browser.close()
    for f in args.tmp.iterdir():
        f.unlink()
    args.tmp.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
