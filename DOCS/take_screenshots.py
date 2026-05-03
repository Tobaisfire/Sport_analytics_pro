"""
Capture Ultimate Dashboard screenshots for DOCS.
Requires: pip install playwright && playwright install chromium
Run Streamlit first:  cd models/ultimate_model && streamlit run app.py --server.port 8501

Usage:
  python take_screenshots.py
  python take_screenshots.py --url http://localhost:8501
"""
from playwright.sync_api import sync_playwright
import time
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--url", default=os.environ.get("STREAMLIT_SCREENSHOT_URL", "http://localhost:8501"))
args = ap.parse_args()

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(out, exist_ok=True)

TAB1 = "Phase 1 — Outcome Prediction"
TAB2 = "Phase 2 — Shot Intelligence"
TAB2AUG = "Phase 2 — Augmented Outcome"
TAB4 = "Model Comparison"
TAB5 = "Leaderboards & Rankings"
TAB6 = "Phase 3 — Narrative prototype (2024)"


def shot(page, name: str, full_page: bool = False, clip=None):
    path = os.path.join(out, name)
    kw: dict = {"path": path, "full_page": full_page}
    if clip:
        kw["clip"] = clip
    page.screenshot(**kw)
    print("saved", name)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1600, "height": 950})
    page.goto(args.url, timeout=60000)
    time.sleep(10)

    try:
        page.get_by_text("Analyze", exact=True).click(timeout=15000)
        time.sleep(8)
    except Exception as e:
        print("Analyze click:", e)

    # Overview (Tab 1 visible)
    shot(page, "01_app_overview.png", full_page=True)

    shot(
        page,
        "02_sidebar_full.png",
        full_page=False,
        clip={"x": 0, "y": 0, "width": 340, "height": 950},
    )

    try:
        page.get_by_text(TAB1).click()
        time.sleep(2)
    except Exception as e:
        print("tab1:", e)
    shot(page, "03_tab1_full.png", full_page=True)

    try:
        page.get_by_text(TAB2).click()
        time.sleep(4)
    except Exception as e:
        print("tab2:", e)
    shot(page, "04_tab2_keyword.png", full_page=True)

    try:
        page.get_by_text(TAB2AUG).click()
        time.sleep(3)
    except Exception as e:
        print("tab2 aug:", e)
    shot(page, "05_tab3_augmented.png", full_page=True)

    try:
        page.get_by_text(TAB4).click()
        time.sleep(3)
    except Exception as e:
        print("tab4:", e)
    shot(page, "06_tab4_model_comparison.png", full_page=True)

    try:
        page.get_by_text(TAB5).click()
        time.sleep(3)
    except Exception as e:
        print("tab5:", e)
    shot(page, "07_tab5_leaderboards.png", full_page=True)

    try:
        page.get_by_text(TAB6).click()
        time.sleep(4)
    except Exception as e:
        print("tab6 phase3:", e)
    shot(page, "08_tab6_phase3_narrative.png", full_page=True)

    browser.close()

print("All screenshots done ->", out)
