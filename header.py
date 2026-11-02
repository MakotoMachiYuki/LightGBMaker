"""Sticky top bar: `LightGBMaker` brand + Home / Classification / About buttons."""

from __future__ import annotations

import streamlit as st

import config as cfg

NAV_ITEMS = [("home", "Home"), ("classification", "Classification"), ("about", "About")]


def _go(view: str) -> None:
    st.session_state.view = view


def render() -> None:
    current = st.session_state.get("view", "home")
    with st.container(
        key="app-header",
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        st.html(
            f"""
            <div class="hdr-brand">
                <div>
                    <div class="hdr-name">{cfg.APP_NAME}</div>
                    <div class="hdr-tag">{cfg.APP_TAGLINE}</div>
                </div>
            </div>
            """,
            width="content",
        )
        with st.container(key="nav-group", horizontal=True, width="content",
                          vertical_alignment="center"):
            for view, label in NAV_ITEMS:
                st.button(
                    label,
                    key=f"nav-{view}",
                    type="primary" if current == view else "secondary",
                    on_click=_go,
                    args=(view,),
                )
