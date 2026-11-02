"""Interactive Folium geospatial map of Indonesia (CARTO Voyager basemap).

Uses Leaflet's HTML5 Canvas renderer (`CanvasPoints`) so thousands of Ookla
network-quality points across Indonesia render smoothly in an embedded iframe.
"""

from __future__ import annotations

import json

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.elements import MacroElement
from folium.template import Template

import config as cfg
import ui

_CLASS_IDX = {c: i for i, c in enumerate(cfg.CLASS_NAMES)}


class CanvasPoints(MacroElement):
    """Draws thousands of points on a single Leaflet HTML5 canvas."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            var cfg = {{ this.payload }};
            var renderer = L.canvas({ padding: 0.5, tolerance: 4 });
            var group = L.featureGroup();

            cfg.points.forEach(function (p) {
                // p = [lat, lon, classIdx, provinceIdx, dl_mbps, ul_mbps, lat_ms]
                var c = cfg.colors[p[2]];
                var layer = L.circleMarker([p[0], p[1]], {
                    renderer: renderer, radius: cfg.radius, color: c, weight: 0,
                    fillColor: c, fillOpacity: cfg.opacity
                });
                layer._p = p;
                group.addLayer(layer);
            });

            group.bindTooltip(function (layer) {
                var p = layer._p;
                return "<b>Quality:</b> " + cfg.classes[p[2]] +
                       " &nbsp;|&nbsp; <b>Province:</b> " + cfg.provinces[p[3]] +
                       "<br><b>DL:</b> " + p[4].toFixed(1) + " Mbps &nbsp; <b>UL:</b> " +
                       p[5].toFixed(1) + " Mbps &nbsp; <b>Latency:</b> " + p[6] + " ms";
            }, { sticky: true, className: "map-tooltip" });
            group.addTo(map);
        })();
        {% endmacro %}
        """
    )

    def __init__(self, df: pd.DataFrame, radius: float = 3.2, opacity: float = 0.78):
        super().__init__()
        self._name = "CanvasPoints"
        prov_col = df["Provinsi"].astype(str) if "Provinsi" in df.columns else pd.Series(["Indonesia"] * len(df))
        provinces = sorted(prov_col.unique().tolist())
        prov_idx = {p: i for i, p in enumerate(provinces)}
        points = [
            [
                round(float(lat), 4),
                round(float(lon), 4),
                _CLASS_IDX.get(str(q), 2),
                prov_idx.get(str(pr), 0),
                round(float(dl) / 1000.0, 1),
                round(float(ul) / 1000.0, 1),
                int(lat_ms),
            ]
            for lat, lon, q, pr, dl, ul, lat_ms in zip(
                df["tile_y"],
                df["tile_x"],
                df[cfg.TARGET],
                prov_col,
                df["avg_d_kbps"],
                df["avg_u_kbps"],
                df["avg_lat_ms"],
            )
        ]
        self.payload = json.dumps(
            {
                "points": points,
                "provinces": provinces,
                "classes": cfg.CLASS_NAMES,
                "colors": [cfg.QUALITY_COLORS[c] for c in cfg.CLASS_NAMES],
                "radius": radius,
                "opacity": opacity,
            },
            separators=(",", ":"),
        )


class MapChrome(MacroElement):
    """Top-right legend + click-to-enable wheel-zoom."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }};
            var legend = L.control({ position: "topright" });
            legend.onAdd = function () {
                var div = L.DomUtil.create("div", "map-legend");
                div.innerHTML = {{ this.legend_html }};
                L.DomEvent.disableClickPropagation(div);
                return div;
            };
            legend.addTo(map);

            map.scrollWheelZoom.disable();
            map.on("click focus", function () { map.scrollWheelZoom.enable(); });
            map.getContainer().addEventListener("mouseleave", function () {
                map.scrollWheelZoom.disable();
            });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, legend_html: str):
        super().__init__()
        self._name = "MapChrome"
        self.legend_html = json.dumps(legend_html)


def _balanced_sample(d: pd.DataFrame, target: int) -> pd.DataFrame:
    """Sample up to `target` rows while preserving minority classes (`Good`, `Great`)."""
    if len(d) <= target:
        return d
    rng = np.random.default_rng(cfg.RANDOM_STATE)
    groups = {c: d[d[cfg.TARGET] == c] for c in cfg.CLASS_NAMES}
    per_class_floor = min(1500, target // 5)
    chosen = []
    remaining_pool = []
    taken = 0
    for c in cfg.CLASS_NAMES:
        g = groups[c]
        if len(g) <= per_class_floor:
            chosen.append(g)
            taken += len(g)
        else:
            idx = rng.choice(len(g), size=per_class_floor, replace=False)
            chosen.append(g.iloc[idx])
            rest_mask = np.ones(len(g), dtype=bool)
            rest_mask[idx] = False
            remaining_pool.append(g.iloc[rest_mask])
            taken += per_class_floor
    need = max(0, target - taken)
    if need > 0 and remaining_pool:
        pool = pd.concat(remaining_pool, ignore_index=False)
        if len(pool) <= need:
            chosen.append(pool)
        else:
            idx = rng.choice(len(pool), size=need, replace=False)
            chosen.append(pool.iloc[idx])
    return pd.concat(chosen, ignore_index=True)


def _legend_html(scope_label: str, shown: int, total: int, counts: dict, drawn: dict) -> str:
    rows = "".join(
        f'<div class="lg-row"><span class="lg-dot" style="background:{cfg.QUALITY_COLORS[c]}"></span>'
        f'<span>{c}</span><span class="lg-n"><b>{drawn.get(c, 0):,}</b> / {counts.get(c, 0):,}</span></div>'
        for c in cfg.CLASS_NAMES
    )
    return (
        f'<div class="lg-title">Quality · {scope_label}</div>{rows}'
        f'<div class="lg-foot">{shown:,} dari {total:,} titik ditampilkan</div>'
    )


def build_map_html(df: pd.DataFrame, year: int | str = "All") -> str:
    """Build Folium map HTML for `df` (filtered by `year` or `'All'`)."""
    if not {"tile_x", "tile_y", cfg.TARGET}.issubset(df.columns) or df.empty:
        return ""
    sub = df
    scope_label = "2023–2026"
    if year != "All" and "year" in df.columns:
        yr_int = int(year)
        sub_y = df[df["year"] == yr_int]
        if not sub_y.empty:
            sub = sub_y
            scope_label = str(yr_int)

    total = len(sub)
    counts = sub[cfg.TARGET].value_counts().to_dict()
    sample = _balanced_sample(sub, cfg.MAP_SAMPLE_PER_YEAR)
    drawn_counts = sample[cfg.TARGET].value_counts().to_dict()

    m = folium.Map(
        location=cfg.MAP_CENTER,
        zoom_start=cfg.MAP_ZOOM,
        tiles=cfg.get_carto_tiles_url(),
        attr=cfg.CARTO_ATTR,
        prefer_canvas=False,
        control_scale=True,
        min_zoom=4,
    )

    font_css = ui.font_face_css()
    if font_css:
        m.get_root().header.add_child(folium.Element(font_css))
    css = cfg.CSS_PATH.read_text(encoding="utf-8") if cfg.CSS_PATH.exists() else ""
    m.get_root().header.add_child(folium.Element(f"<style>{css}</style>"))

    CanvasPoints(sample, radius=3.2, opacity=0.78).add_to(m)
    MapChrome(_legend_html(scope_label, len(sample), total, counts, drawn_counts)).add_to(m)
    return m.get_root().render()


@st.fragment
def render_map_section(df: pd.DataFrame, key_prefix: str = "home") -> None:
    """Render an interactive Indonesia map card with a simple year selector."""
    if df is None or df.empty or not {"tile_x", "tile_y"}.issubset(df.columns):
        return
    years_avail = (
        sorted(int(y) for y in df["year"].dropna().unique())
        if "year" in df.columns
        else []
    )
    options = ["All"] + [str(y) for y in years_avail] if len(years_avail) > 1 else ["All"]
    state_key = f"{key_prefix}_map_year"
    if st.session_state.get(state_key) not in options:
        st.session_state[state_key] = "All"

    if len(options) > 1:
        with st.container(key=f"{key_prefix}-year-bar", horizontal=True, horizontal_alignment="right"):
            st.segmented_control(
                "Tahun Peta",
                options=options,
                key=state_key,
                format_func=lambda x: "Semua Tahun (2023–2026)" if x == "All" else f"Tahun {x}",
                label_visibility="collapsed",
            )

    chosen_year = st.session_state.get(state_key, "All")
    cache_store = st.session_state.setdefault("map_html_cache", {})
    cache_id = f"{id(df)}:{len(df)}:{chosen_year}"
    if cache_id not in cache_store:
        cache_store[cache_id] = build_map_html(df, chosen_year)
    st.iframe(cache_store[cache_id], height=cfg.MAP_HEIGHT)
