"""Capture demo assets (screenshot + GIF frames) from the running app.

Usage:
    python scripts/capture_assets.py [--base-url http://localhost:8501]

Requires: playwright chromium (pip install playwright && playwright install
chromium), and the app running via `streamlit run app.py`.

Produces:
    assets/screenshot_replay.png    full-page replay view
    assets/screenshot_results.png   results tab
    assets/demo.gif                 animated walk-through (Next-trial steps)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS = PROJECT_ROOT / "assets"


def click_button(page, label: str) -> bool:
    """Click a Streamlit button by its visible text; return True if found."""
    target = page.locator(f'button:has-text("{label}")').first
    if target.count() == 0:
        return False
    target.scroll_into_view_if_needed()
    target.click()
    time.sleep(1.0)  # let the fragment/rerun settle
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:8501")
    args = ap.parse_args()
    ASSETS.mkdir(exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(args.base_url, wait_until="networkidle")
        page.wait_for_timeout(6000)  # spinner + data load

        # 1) Initial state
        page.screenshot(path=str(ASSETS / "screenshot_initial.png"), full_page=False)

        # 2) Step through three trials with Next trial
        frames = []
        for step in range(3):
            if not click_button(page, "Next trial"):
                print("WARN: Next trial button not found")
                break
            page.wait_for_timeout(800)
            path = ASSETS / f"_gif_frame_{step}.png"
            page.screenshot(path=str(path))
            frames.append(path)

        # 3) Full replay view screenshot (waveform + cursor + prediction)
        page.screenshot(path=str(ASSETS / "screenshot_replay.png"), full_page=False)

        # 4) Results tab
        results_tab = page.get_by_role("tab", name="Overview").first
        if results_tab.count() == 0:
            # fallback: any element containing the results header text
            page.locator("text=Evaluation results").first.scroll_into_view_if_needed()
        else:
            results_tab.scroll_into_view_if_needed()
            results_tab.click()
        page.wait_for_timeout(1500)
        page.locator("text=Evaluation results").first.scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        page.screenshot(path=str(ASSETS / "screenshot_results.png"), full_page=True)

        browser.close()

    # Build the GIF from frames
    from PIL import Image

    png_frames = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    if png_frames:
        out = ASSETS / "demo.gif"
        png_frames[0].save(
            out, save_all=True, append_images=png_frames[1:], duration=1400, loop=0
        )
        print(f"wrote {out} ({len(png_frames)} frames)")
    for f in frames:
        f.unlink()  # remove intermediate frames
    print("wrote screenshot_initial.png, screenshot_replay.png, screenshot_results.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
