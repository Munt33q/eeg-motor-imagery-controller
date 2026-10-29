"""Interactive debug sweep: click every control, capture any UI exceptions.

Usage:
    python scripts/debug_app_sweep.py [--base-url http://localhost:8501]

Reports the full text of any [data-testid="stException"] elements found.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_exceptions(page) -> list[str]:
    elems = page.locator('[data-testid="stException"]')
    out = []
    for i in range(elems.count()):
        out.append(elems.nth(i).inner_text())
    return out


def alert_boxes(page) -> list[str]:
    elems = page.locator('[data-testid="stAlert"]')
    return [elems.nth(i).inner_text() for i in range(elems.count())]


def click(page, label: str) -> bool:
    btn = page.locator(f'button:has-text("{label}")').first
    if btn.count() == 0:
        print(f"  [MISS] button {label!r} not found")
        return False
    btn.scroll_into_view_if_needed()
    btn.click()
    time.sleep(1.2)
    errs = find_exceptions(page)
    if errs:
        print(f"  [ERROR] after clicking {label!r}:")
        for e in errs:
            print("    " + e[:1500].replace("\n", "\n    "))
        return False
    print(f"  [ok] {label}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8501")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda exc: print(f"  [PAGEERROR] {exc}"))
        page.on(
            "console",
            lambda msg: print(f"  [console.{msg.type}] {msg.text[:300]}")
            if msg.type == "error"
            else None,
        )
        page.goto(args.base_url, wait_until="networkidle")
        page.wait_for_timeout(6000)

        errs = find_exceptions(page)
        print(f"Initial load: {'CLEAN' if not errs else errs}")
        for s in args.subjects:
            print(f"\n=== Subject {s} ===")
            # Open the selectbox dropdown: click, then choose via keyboard
            # (baseweb select internals differ across versions; keyboard is
            # robust). Current value text sits in the stSelectbox div.
            box = page.locator('[data-testid="stSelectbox"]').first
            box.click()  # opens the dropdown listbox
            page.wait_for_timeout(500)
            opts = page.locator('li[data-idx], [role="option"]')
            n_opts = opts.count()
            picked = False
            for i in range(n_opts):
                txt = opts.nth(i).inner_text().strip()
                if txt == str(s):
                    opts.nth(i).click()
                    picked = True
                    break
            if not picked:
                # fallback: keyboard navigation from current value
                print(f"  [WARN] option {s} not listed ({n_opts} options); using keyboard")
                box.press("Escape")
            page.wait_for_timeout(5000)
            errs = find_exceptions(page)
            if errs:
                print(f"  [ERROR] after selecting subject {s}:")
                for e in errs:
                    print("    " + e[:2000].replace("\n", "\n    "))
                continue
            click(page, "Next trial")
            click(page, "Next trial")
            click(page, "Next trial")
            click(page, "Play")
            page.wait_for_timeout(3500)
            errs = find_exceptions(page)
            if errs:
                print("  [ERROR] during Play:")
                for e in errs:
                    print("    " + e[:2000].replace("\n", "\n    "))
            click(page, "Pause")
            # speed slider: click at 75% of the tick bar (stSliderTickBar)
            track = page.locator('[data-testid="stSliderTickBar"]').first
            if track.count():
                box = track.bounding_box()
                if box:
                    page.mouse.click(
                        box["x"] + box["width"] * 0.75, box["y"] + box["height"] / 2
                    )
                    time.sleep(1.2)
                    print("  [ok] speed slider moved to ~2x")
            click(page, "Reset")
            # Results tabs
            for tab_name in ("Overview", "Confusion matrix", "Trial-level predictions"):
                tab = page.get_by_role("tab", name=tab_name).first
                if tab.count():
                    tab.scroll_into_view_if_needed()
                    tab.click()
                    time.sleep(1.0)
                    errs = find_exceptions(page)
                    if errs:
                        print(f"  [ERROR] in tab {tab_name!r}: {errs[0][:800]}")
                    else:
                        print(f"  [ok] tab {tab_name}")
                else:
                    print(f"  [MISS] tab {tab_name!r} not found")

        browser.close()
    print("\nSweep complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
