"""LightGBMaker — Streamlit entry point.

Run from the repository root or from this folder:

    cd LightGBMaker
    streamlit run main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402

st.set_page_config(
    page_title=cfg.APP_NAME,
    page_icon=cfg.APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)

import about  # noqa: E402
import classification  # noqa: E402
import header  # noqa: E402
import home  # noqa: E402
import ui  # noqa: E402


def _init_state() -> None:
    st.session_state.setdefault("view", "home")
    st.session_state.setdefault("pp", None)            # PipelineResult
    st.session_state.setdefault("results", None)       # training results dict
    st.session_state.setdefault("rec", None)           # recommendation dict
    st.session_state.setdefault("charts", {})          # evaluation charts (PNG bytes)
    st.session_state.setdefault("pp_charts", {})       # pre-processing charts
    st.session_state.setdefault("exports", {})         # cached export bytes
    st.session_state.setdefault("dl_cache", {})        # cleaned dataset downloads
    st.session_state.setdefault("primary_choice", "auto")


def _inject_styles() -> None:
    font_css = ui.font_face_css()
    if font_css:
        st.html(font_css)
    if cfg.CSS_PATH.exists():
        st.html(f"<style>{cfg.CSS_PATH.read_text(encoding='utf-8')}</style>")


def main() -> None:
    _init_state()
    _inject_styles()
    header.render()

    view = st.session_state.get("view", "home")
    if view == "classification":
        classification.render()
    elif view == "about":
        about.render()
    else:
        home.render()


if __name__ == "__main__":
    main()
