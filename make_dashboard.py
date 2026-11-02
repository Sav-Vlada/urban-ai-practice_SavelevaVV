#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Собирает один HTML-файл с интерактивной картой, графиками, таблицами и выводами.

Скрипт НИЧЕГО не скачивает и НИЧЕГО не пересчитывает. Он читает готовые файлы
из data_stavropol/ (их создал ноутбук stavropol_pipeline.ipynb) и складывает
из них одну страницу dashboard_stavropol.html.

Запуск:
    .\\.venv_blocksnet\\Scripts\\python.exe make_dashboard.py
"""

from __future__ import annotations

import base64
import html
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from string import Template

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import set_precision

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_stavropol"
RESULTS_DIR = DATA_DIR / "results"
PROCESSED_DIR = DATA_DIR / "processed"
OUT_PATH = PROJECT_DIR / "dashboard.html"

NO_DATA_COLOR = "#e6e6e6"

PALETTES = {
    "orange": ["#ffffcc", "#fd8d3c", "#800026"],
    "blue": ["#f7fbff", "#6baed6", "#08306b"],
    "green": ["#f7fcf5", "#74c476", "#00441b"],
    "purple": ["#fcfbfd", "#9e9ac8", "#3f007d"],
    "grey": ["#f7f7f7", "#bdbdbd", "#252525"],
    "teal": ["#f7fcf0", "#7bccc4", "#084081"],
    "good_bad": ["#1a9850", "#fee090", "#d73027"],
    "bad_good": ["#d73027", "#fee090", "#1a9850"],
}

CAT_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#86bcb6", "#d37295",
    "#a0cbe8", "#ffbe7d", "#8cd17d", "#b6992d",
]

LAND_USE_RU = {
    "residential": "жилое",
    "industrial": "промышленное",
    "business": "деловое",
    "recreation": "рекреационное",
    "agriculture": "сельскохозяйственное",
    "special": "специальное",
    "transport": "транспортное",
}

MORPHO_PARTS_RU = {
    "low-rise": "малоэтажная",
    "mid-rise": "среднеэтажная",
    "high-mid-rise": "повышенной этажности",
    "high-rise": "многоэтажная",
    "tower": "высотная",
    "low-density": "низкоплотная",
    "high-density": "высокоплотная",
    "residential": "жилая",
    "non-residential": "нежилая",
    "mixed-use": "смешанная",
    "model": "типовая",
    "individual": "индивидуальная",
    "block": "квартальная",
    "central": "центральная",
    "modern": "современная",
    "microdistrict": "микрорайонная",
    "special": "специальная",
}


# ---------------------------------------------------------------------------
# Мелкие помощники
# ---------------------------------------------------------------------------

def num(value, digits=0, suffix=""):
    """Красивое число: 12 345 вместо 12345.0."""
    if isinstance(value, str):
        try:
            value = float(value.replace(",", "."))
        except ValueError:
            return value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    text = f"{value:,.{digits}f}".replace(",", "\u00a0").replace(".", ",")
    if text.startswith("-"):
        text = "−" + text[1:]
    return text + suffix


def pct(value, digits=0):
    if isinstance(value, str):
        try:
            value = float(value.replace(",", "."))
        except ValueError:
            return value
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def plural(n, one, few, many):
    """Русское склонение: 1 квартал, 2 квартала, 5 кварталов."""
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def ru_morphotype(text):
    """Переводит английское название морфотипа на русский."""
    if not isinstance(text, str) or not text:
        return "—"
    parts = text.split()
    words = [MORPHO_PARTS_RU.get(part, part) for part in parts]
    return " ".join(words)


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#" + "".join(f"{int(round(c)):02x}" for c in rgb)


def lerp_color(stops, t):
    """Цвет между опорными цветами палитры: t от 0 до 1."""
    t = min(max(t, 0.0), 1.0)
    if len(stops) == 1:
        return stops[0]
    position = t * (len(stops) - 1)
    index = min(int(position), len(stops) - 2)
    local = position - index
    start = hex_to_rgb(stops[index])
    end = hex_to_rgb(stops[index + 1])
    return rgb_to_hex(tuple(start[i] + (end[i] - start[i]) * local for i in range(3)))


def escape(text):
    return html.escape(str(text), quote=True)


# ---------------------------------------------------------------------------
# Какие файлы читаем
# ---------------------------------------------------------------------------

def find_territory_key():
    """Берёт территорию из territory.json, иначе ищет готовые файлы."""
    meta_path = DATA_DIR / "territory.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            key = meta.get("territory_key")
            if key and (RESULTS_DIR / f"{key}_blocks_final.geojson").exists():
                return key, meta
        except json.JSONDecodeError:
            pass

    for candidate in sorted(RESULTS_DIR.glob("*_blocks_final.geojson")):
        key = candidate.name.replace("_blocks_final.geojson", "")
        return key, {"territory_key": key}

    raise SystemExit(
        "Не нашёл файл результатов.\n"
        f"Ожидаю {RESULTS_DIR / '<территория>_blocks_final.geojson'}\n"
        "Сначала выполните ноутбук stavropol_pipeline.ipynb до раздела 12."
    )


def read_optional_csv(path):
    if path.exists():
        return pd.read_csv(path)
    print(f"   ⚠️ нет файла {path.name} — этот блок будет пропущен")
    return None


def load_inputs(key):
    blocks_path = RESULTS_DIR / f"{key}_blocks_final.geojson"
    if not blocks_path.exists():
        raise SystemExit(f"Нет файла {blocks_path}. Сначала выполните раздел 12 ноутбука.")

    blocks = gpd.read_file(blocks_path)
    if blocks.crs is None:
        blocks = blocks.set_crs("EPSG:4326")
    blocks = blocks.to_crs("EPSG:4326")

    for column in ("population", "site_area", "area_ha", "fsi", "gsi", "mxi", "l", "osr"):
        if column in blocks.columns:
            blocks[column] = pd.to_numeric(blocks[column], errors="coerce")
    blocks = blocks.replace([np.inf, -np.inf], np.nan)

    area_km2 = blocks["site_area"] / 1e6
    blocks["population_density"] = np.where(
        area_km2 > 0, blocks["population"] / area_km2, np.nan
    )

    data = {
        "blocks": blocks,
        "metrics": read_optional_csv(RESULTS_DIR / f"{key}_metrics.csv"),
        "provision": read_optional_csv(RESULTS_DIR / f"{key}_provision.csv"),
        "provision_links": read_optional_csv(RESULTS_DIR / f"{key}_provision_links.csv"),
        "area_access": read_optional_csv(RESULTS_DIR / f"{key}_area_accessibility.csv"),
        "spacematrix_clusters": read_optional_csv(PROCESSED_DIR / f"{key}_spacematrix_clusters.csv"),
        "strelka": read_optional_csv(PROCESSED_DIR / f"{key}_strelka.csv"),
    }
    return data


# ---------------------------------------------------------------------------
# Показатели карты
# ---------------------------------------------------------------------------

def build_metric_specs(df):
    """Список слоёв карты: что показывать и как раскрашивать."""
    specs = [
        {
            "key": "density", "column": "population_density", "kind": "continuous",
            "title": "Плотность населения", "unit": "чел/км²", "palette": "orange",
            "digits": 0, "vmin": 0,
            "desc": "Сколько человек живёт на квадратном километре квартала. "
                    "Считано по модельному населению: 30 м² жилой площади на человека.",
        },
        {
            "key": "population", "column": "population", "kind": "continuous",
            "title": "Население квартала", "unit": "чел.", "palette": "purple",
            "digits": 0, "vmin": 0,
            "desc": "Модельное население квартала — сумма населения зданий.",
        },
        {
            "key": "fsi", "column": "fsi", "kind": "continuous",
            "title": "FSI — плотность застройки", "unit": "м² этажей / м² квартала",
            "palette": "blue", "digits": 2, "vmin": 0,
            "desc": "Отношение площади всех этажей зданий к площади квартала. "
                    "0,1 — одноэтажная застройка, 1,0 — пять этажей почти на всей площади.",
        },
        {
            "key": "gsi", "column": "gsi", "kind": "continuous",
            "title": "GSI — доля застроенной земли", "unit": "доля",
            "palette": "teal", "digits": 2, "vmin": 0, "vmax": 1,
            "desc": "Какая часть земли квартала занята зданиями (пятна застройки).",
        },
        {
            "key": "mxi", "column": "mxi", "kind": "continuous",
            "title": "MXI — доля жилых площадей", "unit": "доля",
            "palette": "green", "digits": 2, "vmin": 0, "vmax": 1,
            "desc": "Какая часть площади этажей жилая. 1,0 — только жильё, 0,0 — только нежилые функции.",
        },
        {
            "key": "l", "column": "l", "kind": "continuous",
            "title": "L — средняя этажность", "unit": "этажей",
            "palette": "grey", "digits": 1, "vmin": 0,
            "desc": "Средневзвешенная этажность застройки квартала. Где в OSM не указана "
                    "этажность, принят 1 этаж — поэтому низкие значения могут быть занижены.",
        },
        {
            "key": "osr", "column": "osr", "kind": "continuous",
            "title": "OSR — открытая земля на 1 м² этажей", "unit": "м² / м²",
            "palette": "teal", "digits": 1, "vmin": 0,
            "desc": "Сколько открытой (незастроенной) земли приходится на квадратный метр "
                    "площади этажей. Больше — свободнее. Для кварталов без зданий значения нет.",
        },
        {
            "key": "provision", "column": "provision_strong", "kind": "continuous",
            "title": "Обеспеченность школами (сильная конкуренция)", "unit": "доля спроса",
            "palette": "bad_good", "digits": 2, "vmin": 0, "vmax": 1,
            "desc": "Какая часть потребности квартала в школьных местах закрыта "
                    "ближайшими школами с учётом конкуренции соседей. 1,0 — потребность закрыта.",
        },
        {
            "key": "acc", "column": "area_accessibility_cap60", "kind": "continuous",
            "title": "Доступность территории", "unit": "минут",
            "palette": "good_bad", "digits": 0, "vmin": None,
            "desc": "Средневзвешенное время пешей доступности до всех кварталов города "
                    "(вес — площадь квартала). Недостижимые пары заменены на 60 минут. "
                    "Меньше — лучше.",
        },
        {
            "key": "land_use", "column": "land_use", "kind": "categorical",
            "title": "Землепользование квартала", "palette": None,
            "desc": "Преобладающая функция квартала по данным OSM: жилая, промышленная, "
                    "деловая и т. д.",
        },
        {
            "key": "strelka", "column": "strelka_morphotype", "kind": "categorical",
            "title": "Морфотип Strelka", "palette": None,
            "desc": "Тип застройки по классификации Strelka: этажность, плотность и функция. "
                    "Серые кварталы — без зданий, тип не определён.",
        },
        {
            "key": "spacematrix", "column": "spacematrix_morphotype", "kind": "categorical",
            "title": "Морфотип Spacematrix", "palette": None,
            "desc": "Второй взгляд на тип застройки — по сочетанию этажности (L), плотности (FSI) "
                    "и доли жилья (MXI).",
        },
        {
            "key": "provision_weak", "column": "provision_weak", "kind": "continuous",
            "title": "Обеспеченность школами (все места)", "unit": "доля спроса",
            "palette": "bad_good", "digits": 2, "vmin": 0, "vmax": 1,
            "desc": "Какая часть потребности закрыта всеми доступными местами, включая далёкие школы.",
        },
        {
            "key": "acc_raw", "column": "area_accessibility_raw", "kind": "continuous",
            "title": "Доступность территории (raw)", "unit": "минут",
            "palette": "good_bad", "digits": 0, "vmin": None,
            "desc": "Средневзвешенное время до всех кварталов без замены inf; недостижимые пары остаются пустыми.",
        },
        {
            "key": "buildings", "column": "building_count", "kind": "continuous",
            "title": "Число зданий", "unit": "шт.", "palette": "blue",
            "digits": 0, "vmin": 0,
            "desc": "Сколько зданий попало в квартал после очистки данных.",
        },
        {
            "key": "footprint", "column": "footprint_area", "kind": "continuous",
            "title": "Площадь застройки", "unit": "м²", "palette": "orange",
            "digits": 0, "vmin": 0,
            "desc": "Сумма площадей пятен застройки: сколько земли занято зданиями.",
        },
        {
            "key": "floor_area", "column": "build_floor_area", "kind": "continuous",
            "title": "Площадь этажей", "unit": "м²", "palette": "purple",
            "digits": 0, "vmin": 0,
            "desc": "Сумма площадей всех этажей всех зданий квартала.",
        },
        {
            "key": "living", "column": "living_area", "kind": "continuous",
            "title": "Жилая площадь", "unit": "м²", "palette": "green",
            "digits": 0, "vmin": 0,
            "desc": "Жилая площадь зданий: 90 % площади этажей жилых домов.",
        },
        {
            "key": "nonliving", "column": "non_living_area", "kind": "continuous",
            "title": "Нежилая площадь", "unit": "м²", "palette": "teal",
            "digits": 0, "vmin": 0,
            "desc": "Нежилая площадь: офисы, торговля, склады и прочие помещения.",
        },
        {
            "key": "share_living", "column": "share_living", "kind": "continuous",
            "title": "Жилая этажность на пятно", "unit": "доля", "palette": "green",
            "digits": 2, "vmin": 0,
            "desc": "Сколько этажей жилой части приходится на 1 м² пятна застройки.",
        },
        {
            "key": "share_nonliving", "column": "share_non_living", "kind": "continuous",
            "title": "Нежилая этажность на пятно", "unit": "доля", "palette": "teal",
            "digits": 2, "vmin": 0,
            "desc": "Сколько этажей нежилой части приходится на 1 м² пятна застройки.",
        },
        {
            "key": "schools_n", "column": "school_count", "kind": "continuous",
            "title": "Число школ", "unit": "шт.", "palette": "bad_good",
            "digits": 0, "vmin": 0,
            "desc": "Сколько школ расположено в квартале по данным OpenStreetMap.",
        },
        {
            "key": "seats_n", "column": "school_capacity", "kind": "continuous",
            "title": "Вместимость школ", "unit": "мест", "palette": "bad_good",
            "digits": 0, "vmin": 0,
            "desc": "Суммарная вместимость школ квартала; при пропуске в OSM принято 250 мест.",
        },
        {
            "key": "demand_n", "column": "demand", "kind": "continuous",
            "title": "Потребность в школьных местах", "unit": "мест", "palette": "bad_good",
            "digits": 0, "vmin": 0,
            "desc": "Нормативная потребность: население ÷ 1000 × 120 мест, округлённая вверх.",
        },
        {
            "key": "share", "column": "share", "kind": "continuous",
            "title": "Доля зоны в квартале", "unit": "доля", "palette": "grey",
            "digits": 2, "vmin": 0, "vmax": 1,
            "desc": "Какая часть площади квартала покрыта зоной, по которой определён тип землепользования.",
        },
    ]

    specs = [spec for spec in specs if spec["column"] in df.columns]

    for spec in specs:
        if spec["kind"] == "continuous":
            values = pd.to_numeric(df[spec["column"]], errors="coerce").replace([np.inf, -np.inf], np.nan)
            finite = values.dropna()
            if spec.get("vmin") is None:
                spec["vmin"] = float(np.floor(finite.min())) if len(finite) else 0.0
            if not spec.get("vmax"):
                if len(finite):
                    vmax = float(finite.quantile(0.95))
                    if vmax <= spec["vmin"]:
                        vmax = float(finite.max())
                    spec["vmax"] = max(vmax, spec["vmin"] + 1e-9)
                else:
                    spec["vmax"] = spec["vmin"] + 1.0
            spec["stops"] = PALETTES[spec["palette"]]
        else:
            categories = sorted({str(v) for v in df[spec["column"]].dropna().unique()})
            spec["categories"] = categories
            spec["colors"] = {cat: CAT_COLORS[i % len(CAT_COLORS)] for i, cat in enumerate(categories)}

    return specs


def metric_color(spec, value):
    if spec["kind"] == "categorical":
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return NO_DATA_COLOR
        return spec["colors"].get(str(value), NO_DATA_COLOR)

    if value is None or (isinstance(value, float) and not math.isfinite(float(value))):
        return NO_DATA_COLOR
    span = spec["vmax"] - spec["vmin"]
    if span <= 0:
        return lerp_color(spec["stops"], 1.0)
    return lerp_color(spec["stops"], (float(value) - spec["vmin"]) / span)


def build_legend(spec):
    """Пять опорных цветов с подписями для легенды карты."""
    if spec["kind"] == "categorical":
        items = [
            {"color": spec["colors"][cat], "label": category_label(spec["key"], cat)}
            for cat in spec["categories"]
        ]
        items.append({"color": NO_DATA_COLOR, "label": "нет данных"})
        return items

    items = []
    for i in range(5):
        t = i / 4
        value = spec["vmin"] + t * (spec["vmax"] - spec["vmin"])
        if spec["key"] == "acc":
            label = num(value, 0, " мин")
        elif spec["unit"] == "доля":
            label = num(value, 2)
        else:
            label = num(value, spec["digits"])
        items.append({"color": lerp_color(spec["stops"], t), "label": label})
    items.append({"color": NO_DATA_COLOR, "label": "нет данных"})
    return items


def category_label(key, value):
    if key == "land_use":
        return LAND_USE_RU.get(value, value)
    return ru_morphotype(value)


# ---------------------------------------------------------------------------
# Карта
# ---------------------------------------------------------------------------

def build_tooltip(row, key):
    lines = [f"<b>Квартал {int(row['block_id'])}</b>"]
    if pd.notna(row.get("land_use")):
        lines.append(f"Землепользование: {LAND_USE_RU.get(row['land_use'], row['land_use'])}")
    if pd.notna(row.get("population")) and row["population"] > 0:
        lines.append(f"Население: {num(row['population'], 0)} чел.")
    if pd.notna(row.get("population_density")) and row["population_density"] > 0:
        lines.append(f"Плотность: {num(row['population_density'], 0)} чел/км²")
    if pd.notna(row.get("area_ha")):
        lines.append(f"Площадь: {num(row['area_ha'], 1)} га")
    if pd.notna(row.get("building_count")):
        lines.append(f"Зданий: {num(row['building_count'])}")
    if pd.notna(row.get("build_floor_area")):
        lines.append(f"Площадь этажей: {num(row['build_floor_area'])} м²")
    density = []
    if pd.notna(row.get("fsi")):
        density.append(f"FSI {num(row['fsi'], 2)}")
    if pd.notna(row.get("gsi")):
        density.append(f"GSI {num(row['gsi'], 2)}")
    if pd.notna(row.get("l")):
        density.append(f"L {num(row['l'], 1)}")
    if density:
        lines.append(" · ".join(density))
    morph = ru_morphotype(row.get("strelka_morphotype"))
    if morph != "—":
        lines.append(f"Морфотип: {morph}")
    if pd.notna(row.get("provision_strong")):
        lines.append(f"Обеспеченность школами: {pct(row['provision_strong'])}")
    if pd.notna(row.get("provision_weak")):
        lines.append(f"Обеспеченность всеми местами: {pct(row['provision_weak'])}")
    if pd.notna(row.get("school_count")):
        seats = num(row.get("school_capacity", 0))
        need = num(row.get("demand", 0))
        lines.append(f"Школ: {num(row['school_count'])} · мест: {seats} · нужно: {need}")
    if pd.notna(row.get("area_accessibility_cap60")):
        lines.append(f"Доступность: {num(row['area_accessibility_cap60'], 0)} мин")
    return "<br>".join(lines)


def build_map_payload(blocks, specs):
    """GeoJSON с готовыми цветами для каждого слоя и подсказкой при наведении."""
    work = blocks.copy()
    work["geometry"] = work.geometry.apply(lambda geom: set_precision(geom, 0.00001))
    geo = json.loads(work.to_json(drop_id=True))

    by_id = {int(row["block_id"]): row for _, row in work.iterrows()} if "block_id" in work else {}

    features = []
    for feature in geo["features"]:
        block_id = feature["properties"].get("block_id")
        row = by_id.get(int(block_id)) if block_id is not None else None
        properties = {"id": int(block_id) if block_id is not None else None}
        if row is not None:
            properties["tip"] = build_tooltip(row, "city")
        else:
            properties["tip"] = f"Квартал {block_id}"
        for spec in specs:
            value = None if row is None else row.get(spec["column"])
            if isinstance(value, (np.floating, np.integer)):
                value = float(value)
            properties[f"c_{spec['key']}"] = metric_color(spec, value)
        feature["properties"] = properties
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


def map_metrics_json(specs):
    payload = []
    for spec in specs:
        payload.append({
            "key": spec["key"],
            "title": spec["title"],
            "desc": spec["desc"],
            "legend": build_legend(spec),
        })
    return payload


# ---------------------------------------------------------------------------
# Графики в SVG (без внешних библиотек)
# ---------------------------------------------------------------------------

def svg_histogram(values, title, xlabel, bins=18, width=520, height=280,
                  color="#4e79a7", fmt="{:.2f}", note=None):
    values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    left, right, top, bottom = 52, 14, 34, 52
    plot_w, plot_h = width - left - right, height - top - bottom

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">',
        f'<text x="{left}" y="20" class="chart-title">{escape(title)}</text>',
    ]
    if values.empty:
        parts.append(f'<text x="{left}" y="{top + plot_h / 2}" class="chart-empty">нет данных</text>')
        parts.append("</svg>")
        return "".join(parts)

    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts.max(), 1)
    vmin, vmax = float(edges[0]), float(edges[-1])
    span = vmax - vmin or 1.0

    for i in range(4):
        y = top + plot_h * i / 3
        count = max_count * (3 - i) / 3
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" class="axis" text-anchor="end">{num(count, 0)}</text>'
        )

    bar_w = plot_w / len(counts)
    for i, count in enumerate(counts):
        if count == 0:
            continue
        bar_h = plot_h * count / max_count
        x = left + i * bar_w + 1
        y = top + plot_h - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 2, 1):.1f}" '
                     f'height="{bar_h:.1f}" fill="{color}" rx="1.5"/>')

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis-line"/>')
    for frac in (0, 0.5, 1):
        x = left + plot_w * frac
        anchor = "start" if frac == 0 else ("end" if frac == 1 else "middle")
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" class="axis" '
                     f'text-anchor="{anchor}">{fmt.format(vmin + span * frac).replace(".", ",")}</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 12}" class="axis-title" '
                 f'text-anchor="middle">{escape(xlabel)}</text>')
    if note:
        parts.append(f'<text x="{left}" y="{top + 12}" class="chart-note">{escape(note)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_bars(items, title, width=520, row_h=26, color="#4e79a7", fmt=None, note=None):
    """Горизонтальные столбики: items = [(подпись, значение, цвет|None), ...]."""
    items = [(item[0], item[1], item[2] if len(item) > 2 and item[2] else color)
             for item in items]
    left, right, top = 8, 96, 34
    label_w = 190
    plot_left = left + label_w
    plot_w = width - plot_left - right
    height = top + row_h * max(len(items), 1) + 18

    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">',
        f'<text x="{left}" y="20" class="chart-title">{escape(title)}</text>',
    ]
    if note:
        parts.append(f'<text x="{left}" y="{top - 6}" class="chart-note">{escape(note)}</text>')
    if not items:
        parts.append(f'<text x="{left}" y="{top + 16}" class="chart-empty">нет данных</text>')
        parts.append("</svg>")
        return "".join(parts)

    max_value = max(value for _, value, _ in items) or 1
    for i, (label, value, bar_color) in enumerate(items):
        y = top + i * row_h
        bar_w = plot_w * value / max_value
        text = f"{fmt(value)}" if fmt else num(value, 0)
        parts.append(
            f'<text x="{plot_left - 8}" y="{y + row_h / 2 + 4:.1f}" class="axis" '
            f'text-anchor="end">{escape(label[:34])}</text>'
        )
        parts.append(
            f'<rect x="{plot_left}" y="{y + 4:.1f}" width="{max(bar_w, 1):.1f}" '
            f'height="{row_h - 10}" fill="{bar_color}" rx="2"/>'
        )
        parts.append(
            f'<text x="{plot_left + bar_w + 6:.1f}" y="{y + row_h / 2 + 4:.1f}" class="axis">{escape(text)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Таблицы
# ---------------------------------------------------------------------------

def df_to_table(frame, classes="tbl"):
    return frame.to_html(index=False, classes=classes, border=0, escape=True, na_rep="—")


def summary_table(stats):
    rows = [
        ("Территория", stats["territory_name"], "из territory.json"),
        ("Кварталов", num(stats["n_blocks"]), "нарезка территории по дорогам, воде и железным дорогам"),
        ("Площадь, км²", num(stats["area_km2"], 1), "сумма площадей кварталов"),
        ("Население (модельное), чел.", num(stats["population"]), "30 м² жилой площади на человека"),
        ("Плотность населения, чел/км²", num(stats["density"]), "население / площадь территории"),
        ("Зданий, шт.", num(stats["buildings"]), "после очистки и заполнения пропусков"),
        ("Школ, шт.", num(stats["schools"]), "из OpenStreetMap (`amenity=school`)"),
        ("Школьных мест, шт.", num(stats["seats"]), "вместимость из OSM, при пропуске — 250 мест"),
        ("Потребность в местах, шт.", num(stats["demand"]), "120 мест на 1000 жителей"),
        ("Обеспеченность местами (город в целом)", pct(stats["capacity_ratio"]), "школьные места / потребность"),
        ("Медиана FSI (застроенные кварталы)", num(stats["median_fsi"], 2), "только кварталы с застройкой"),
        ("Медиана GSI", num(stats["median_gsi"], 2), "доля застроенной земли"),
        ("Медиана этажности L", num(stats["median_l"], 1), "по кварталам с застройкой"),
        ("Медиана доступности, мин", num(stats["median_acc"], 1), "area_accessibility, недостижимое = 60 мин"),
    ]
    frame = pd.DataFrame(rows, columns=["Показатель", "Значение", "Как считалось"])
    return df_to_table(frame)


def land_use_table(blocks):
    frame = blocks.copy()
    frame["land_use_ru"] = frame["land_use"].map(lambda v: LAND_USE_RU.get(v, v)).fillna("нет данных")
    grouped = frame.groupby("land_use_ru").agg(
        кварталов=("block_id", "size"),
        площадь_км2=("site_area", lambda s: s.sum() / 1e6),
        население=("population", "sum"),
        медиана_FSI=("fsi", "median"),
    ).reset_index()
    total_area = grouped["площадь_км2"].sum() or 1
    grouped["доля_площади"] = grouped["площадь_км2"] / total_area
    grouped = grouped.sort_values("площадь_км2", ascending=False)
    grouped.columns = ["Землепользование", "Кварталов", "Площадь, км²", "Население, чел.",
                       "Медиана FSI", "Доля площади"]
    grouped["Площадь, км²"] = grouped["Площадь, км²"].map(lambda v: num(v, 1))
    grouped["Население, чел."] = grouped["Население, чел."].map(lambda v: num(v))
    grouped["Медиана FSI"] = grouped["Медиана FSI"].map(lambda v: num(v, 2))
    grouped["Доля площади"] = grouped["Доля площади"].map(lambda v: pct(v, 0))
    return df_to_table(grouped), frame


def morphotype_table(blocks, column, title_ru):
    frame = blocks.dropna(subset=[column]).copy()
    if frame.empty:
        return None
    frame["label"] = frame[column].map(ru_morphotype)
    grouped = frame.groupby("label").agg(
        кварталов=("block_id", "size"),
        площадь_км2=("site_area", lambda s: s.sum() / 1e6),
        население=("population", "sum"),
        медиана_FSI=("fsi", "median"),
        медиана_L=("l", "median"),
    ).reset_index()
    grouped["доля"] = grouped["кварталов"] / len(blocks)
    grouped = grouped.sort_values("кварталов", ascending=False)
    grouped.columns = [title_ru, "Кварталов", "Площадь, км²", "Население, чел.",
                       "Медиана FSI", "Медиана L", "Доля кварталов"]
    grouped["Площадь, км²"] = grouped["Площадь, км²"].map(lambda v: num(v, 1))
    grouped["Население, чел."] = grouped["Население, чел."].map(lambda v: num(v))
    grouped["Медиана FSI"] = grouped["Медиана FSI"].map(lambda v: num(v, 2))
    grouped["Медиана L"] = grouped["Медиана L"].map(lambda v: num(v, 1))
    grouped["Доля кварталов"] = grouped["Доля кварталов"].map(lambda v: pct(v, 1))
    return df_to_table(grouped)


def provision_table(blocks):
    frame = blocks.copy()
    frame = frame[frame["demand"].fillna(0) > 0]
    if frame.empty:
        return None, None

    frame["allocated"] = frame["demand"].fillna(0) * frame["provision_strong"].fillna(0)
    values = frame["provision_strong"].fillna(0)
    buckets = pd.cut(
        values,
        bins=[-1e-9, 1e-9, 0.25, 0.5, 0.75, 1.0, np.inf],
        labels=["0%", "до 25%", "25–50%", "50–75%", "75–100%", "больше 100%"],
    )
    grouped = frame.assign(bucket=buckets).groupby("bucket", observed=False).agg(
        кварталов=("block_id", "size"),
        потребность=("demand", "sum"),
        места_рядом=("allocated", "sum"),
    ).reset_index()
    grouped["доля"] = grouped["кварталов"] / len(frame)
    grouped.columns = ["Обеспеченность", "Кварталов", "Потребность, мест", "Распределено мест",
                       "Доля кварталов"]
    grouped["Потребность, мест"] = grouped["Потребность, мест"].map(lambda v: num(v))
    grouped["Распределено мест"] = grouped["Распределено мест"].map(lambda v: num(v))
    grouped["Доля кварталов"] = grouped["Доля кварталов"].map(lambda v: pct(v, 1))

    worst = frame.sort_values(["provision_strong", "demand"], ascending=[True, False]).head(10).copy()
    worst["strelka_morphotype"] = worst["strelka_morphotype"].map(ru_morphotype)
    worst["land_use"] = worst["land_use"].map(lambda v: LAND_USE_RU.get(v, v))
    worst = worst[[
        "block_id", "land_use", "population", "demand", "provision_strong",
        "school_count", "area_accessibility_cap60",
    ]]
    worst.columns = ["Квартал", "Землепользование", "Население, чел.", "Потребность, мест",
                     "Обеспеченность", "Школ в квартале", "Доступность, мин"]
    worst["Население, чел."] = worst["Население, чел."].map(lambda v: num(v))
    worst["Потребность, мест"] = worst["Потребность, мест"].map(lambda v: num(v))
    worst["Обеспеченность"] = worst["Обеспеченность"].map(lambda v: pct(v))
    worst["Школ в квартале"] = worst["Школ в квартале"].map(lambda v: num(v))
    worst["Доступность, мин"] = worst["Доступность, мин"].map(lambda v: num(v, 0))
    return df_to_table(grouped), df_to_table(worst)


def top_blocks_table(blocks, column, title, ascending=False, digits=0, limit=10):
    frame = blocks.dropna(subset=[column]).sort_values(column, ascending=ascending).head(limit).copy()
    if frame.empty:
        return None
    frame["land_use"] = frame["land_use"].map(lambda v: LAND_USE_RU.get(v, v))
    frame["strelka_morphotype"] = frame["strelka_morphotype"].map(ru_morphotype)
    base_columns = [
        "block_id", "land_use", "strelka_morphotype", "population", "area_ha",
        "fsi", "l", "provision_strong", "area_accessibility_cap60",
    ]
    selected = list(dict.fromkeys(base_columns + [column]))
    frame = frame[selected]

    rename = {
        "block_id": "Квартал", "land_use": "Землепользование",
        "strelka_morphotype": "Морфотип", "population": "Население, чел.",
        "area_ha": "Площадь, га", "fsi": "FSI", "l": "L",
        "provision_strong": "Обеспеченность",
        "area_accessibility_cap60": "Доступность, мин",
    }
    extra = column not in base_columns
    if extra:
        rename[column] = title
    frame = frame.rename(columns=rename)

    frame["Население, чел."] = frame["Население, чел."].map(lambda v: num(v))
    frame["Площадь, га"] = frame["Площадь, га"].map(lambda v: num(v, 1))
    frame["FSI"] = frame["FSI"].map(lambda v: num(v, 2))
    frame["L"] = frame["L"].map(lambda v: num(v, 1))
    frame["Обеспеченность"] = frame["Обеспеченность"].map(lambda v: pct(v))
    frame["Доступность, мин"] = frame["Доступность, мин"].map(lambda v: num(v, 0))
    if extra:
        frame[title] = frame[title].map(lambda v: num(v, digits))
    return df_to_table(frame)


PROFILE_COLUMNS = [
    ("site_area", "Площадь квартала", "м²", 0),
    ("area_ha", "Площадь квартала", "га", 1),
    ("share", "Доля зоны", "доля", 2),
    ("population", "Население", "чел.", 0),
    ("building_count", "Зданий", "шт.", 0),
    ("footprint_area", "Площадь застройки", "м²", 0),
    ("build_floor_area", "Площадь этажей", "м²", 0),
    ("living_area", "Жилая площадь", "м²", 0),
    ("non_living_area", "Нежилая площадь", "м²", 0),
    ("school_count", "Школ", "шт.", 0),
    ("school_capacity", "Вместимость школ", "мест", 0),
    ("demand", "Потребность в местах", "мест", 0),
    ("provision_strong", "Обеспеченность (15 мин)", "доля", 2),
    ("provision_weak", "Обеспеченность (все места)", "доля", 2),
    ("fsi", "FSI", "м²/м²", 2),
    ("gsi", "GSI", "доля", 2),
    ("mxi", "MXI", "доля", 2),
    ("l", "L", "этажей", 1),
    ("osr", "OSR", "м²/м²", 1),
    ("share_living", "Жилая этажность на пятно", "доля", 2),
    ("share_non_living", "Нежилая этажность на пятно", "доля", 2),
    ("spacematrix_cluster", "Кластер Spacematrix", "номер", 0),
    ("area_accessibility_raw", "Доступность (raw)", "мин", 1),
    ("area_accessibility_cap60", "Доступность (cap 60)", "мин", 1),
]


def indicator_profile_table(blocks):
    rows = []
    for column, title, unit, digits in PROFILE_COLUMNS:
        if column not in blocks.columns:
            continue
        values = pd.to_numeric(blocks[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite = values.dropna()
        if finite.empty:
            continue
        rows.append({
            "Показатель": title,
            "Единица": unit,
            "Кварталов": num(finite.size),
            "Медиана": num(finite.median(), digits),
            "Среднее": num(finite.mean(), digits),
            "Минимум": num(finite.min(), digits),
            "Максимум": num(finite.max(), digits),
        })
    return df_to_table(pd.DataFrame(rows))


def spacematrix_table(clusters, blocks):
    if clusters is None or clusters.empty:
        return None
    counts = blocks.dropna(subset=["spacematrix_cluster"]).groupby("spacematrix_cluster").size()
    frame = clusters.copy()
    frame["кварталов"] = frame["cluster"].map(counts).fillna(0).astype(int)
    frame["morphotype"] = frame["morphotype"].map(ru_morphotype)
    frame = frame[["cluster", "morphotype", "l", "fsi", "mxi", "кварталов"]]
    frame.columns = ["Кластер", "Морфотип", "Медиана L", "Медиана FSI", "Медиана MXI", "Кварталов"]
    frame["Медиана L"] = frame["Медиана L"].map(lambda v: num(v, 2))
    frame["Медиана FSI"] = frame["Медиана FSI"].map(lambda v: num(v, 2))
    frame["Медиана MXI"] = frame["Медиана MXI"].map(lambda v: num(v, 2))
    return df_to_table(frame.sort_values("Кварталов", ascending=False))


# ---------------------------------------------------------------------------
# Статистика и выводы
# ---------------------------------------------------------------------------

def compute_stats(blocks, meta):
    built = blocks[blocks["fsi"].fillna(0) > 0]
    population = float(blocks["population"].sum())
    area_km2 = float(blocks["site_area"].sum() / 1e6)
    seats = float(blocks["school_capacity"].fillna(0).sum()) if "school_capacity" in blocks else 0.0
    demand = float(blocks["demand"].fillna(0).sum()) if "demand" in blocks else 0.0

    stats = {
        "territory_name": meta.get("territory_name") or meta.get("territory_key", "территория"),
        "n_blocks": int(len(blocks)),
        "area_km2": area_km2,
        "population": population,
        "density": population / area_km2 if area_km2 else None,
        "buildings": float(blocks["building_count"].fillna(0).sum()) if "building_count" in blocks else float("nan"),
        "schools": float(blocks["school_count"].fillna(0).sum()) if "school_count" in blocks else float("nan"),
        "seats": seats,
        "demand": demand,
        "capacity_ratio": seats / demand if demand else None,
        "median_fsi": float(built["fsi"].median()) if len(built) else float("nan"),
        "median_gsi": float(blocks["gsi"].median()) if "gsi" in blocks else float("nan"),
        "median_l": float(built["l"].median()) if len(built) else float("nan"),
        "median_acc": float(blocks["area_accessibility_cap60"].median())
        if "area_accessibility_cap60" in blocks else float("nan"),
        "built_blocks": int(len(built)),
        "built_share": len(built) / len(blocks) if len(blocks) else 0.0,
        "below_norm_blocks": int(((blocks["demand"].fillna(0) > 0) & (blocks["provision_strong"].fillna(0) <= 0)).sum())
        if "provision_strong" in blocks else 0,
        "demand_blocks": int((blocks["demand"].fillna(0) > 0).sum()) if "demand" in blocks else 0,
    }
    return stats


def build_conclusions(stats, blocks):
    built = blocks[blocks["fsi"].fillna(0) > 0]
    paragraphs = []

    paragraphs.append(
        f"<p>Территория охватывает <b>{num(stats['area_km2'], 1)} км²</b> и "
        f"<b>{num(stats['n_blocks'])} {plural(stats['n_blocks'], 'квартал', 'квартала', 'кварталов')}</b>. "
        f"Модельное население — <b>{num(stats['population'])} человек</b>, это около "
        f"<b>{num(stats['density'])} человек на км²</b>. Средняя плотность по городу "
        "складывается из очень разных кварталов: в жилых она в разы выше, "
        "в сельскохозяйственных и промышленных — почти нулевая.</p>"
    )

    if stats["built_blocks"]:
        paragraphs.append(
            f"<p>Застройка есть в <b>{stats['built_blocks']} "
            f"{plural(stats['built_blocks'], 'квартале', 'кварталах', 'кварталах')}</b> "
            f"({pct(stats['built_share'], 1)} от всех). Медианный FSI — "
            f"<b>{num(stats['median_fsi'], 2)}</b>, медианный GSI — "
            f"<b>{num(stats['median_gsi'], 2)}</b>, медианное значение L — "
            f"<b>{num(stats['median_l'], 1)}</b>. Низкая этажность — "
            "во многом следствие данных: там, где в OpenStreetMap не указано число этажей, "
            "ноутбук принимает 1 этаж. Поэтому FSI и L скорее занижены, чем завышены.</p>"
        )

    land_use = blocks.dropna(subset=["land_use"])
    if not land_use.empty:
        area_by_use = land_use.groupby("land_use")["site_area"].sum().sort_values(ascending=False)
        top_use = area_by_use.index[0]
        top_share = area_by_use.iloc[0] / area_by_use.sum()
        residential_share = area_by_use.get("residential", 0) / area_by_use.sum()
        population = blocks.loc[blocks["land_use"] == "residential", "population"].sum()
        population_share = population / stats["population"] if stats["population"] else None
        paragraphs.append(
            f"<p>По площади преобладает <b>{LAND_USE_RU.get(top_use, top_use)} "
            f"землепользование</b> — {pct(top_share, 0)} территории. На жилые кварталы "
            f"приходится <b>{pct(residential_share, 0)}</b> площади, и в них живёт "
            f"<b>{pct(population_share, 0)}</b> модельного населения. Остальное — "
            "промышленность, сельхозземли, рекреация и деловые кварталы.</p>"
        )

    if stats["schools"] == stats["schools"] and stats["demand"]:
        paragraphs.append(
            f"<p>Школ — <b>{num(stats['schools'])}</b>, мест — <b>{num(stats['seats'])}</b>, "
            f"потребность по нормативу 120 мест на 1000 жителей — "
            f"<b>{num(stats['demand'])} мест</b>. Если считать «в целом по городу», "
            f"места закрывают <b>{pct(stats['capacity_ratio'])}</b> потребности. "
            f"Но места распределены неравномерно: в <b>{stats['below_norm_blocks']}</b> "
            f"{plural(stats['below_norm_blocks'], 'квартале', 'кварталах', 'кварталах')} "
            f"из {stats['demand_blocks']} кварталов с потребностью ближайшие школы "
            "не дают ни одного места при сильной конкуренции соседей. "
            "Это главный содержательный вывод по услугам.</p>"
        )

    if stats["median_acc"] == stats["median_acc"]:
        paragraphs.append(
            f"<p>Медианная доступность территории — <b>{num(stats['median_acc'], 1)} минуты</b>. "
            "Показатель считает средневзвешенное время до всех кварталов города, поэтому "
            "он всегда заметно больше «15 минут до школы»: это характеристика положения "
            "квартала в городе, а не похода в конкретное место. Чем ближе к центру и чем "
            "плотнее сеть улиц, тем меньше значение.</p>"
        )

    if "strelka_morphotype" in blocks:
        morph = blocks["strelka_morphotype"].dropna()
        if not morph.empty:
            top_morph = morph.value_counts()
            paragraphs.append(
                f"<p>Самый частый морфотип Strelka — <b>{ru_morphotype(top_morph.index[0])}</b>: "
                f"{num(top_morph.iloc[0])} "
                f"{plural(top_morph.iloc[0], 'квартал', 'квартала', 'кварталов')} "
                f"({pct(top_morph.iloc[0] / stats['n_blocks'], 0)}). "
                f"У {num(stats['n_blocks'] - len(morph))} кварталов тип не определён — "
                "в них нет зданий (вода, пустыри, сельхозземли).</p>"
            )

    paragraphs.append(
        "<p class=\"warn\">Это учебное исследование, а не официальная статистика. "
        "Население восстановлено по площади жилья, вместимость школ — по нормативу и данным OSM, "
        "границы кварталов проведены по дорогам, воде и железным дорогам. "
        "Числа показывают порядок величин и различия между районами, но не заменяют отчётность.</p>"
    )
    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Сборка страницы
# ---------------------------------------------------------------------------

STYLE = """
:root {
  --bg: #f4f6f9; --card: #ffffff; --ink: #1c2430; --muted: #6b7787;
  --line: #e2e7ee; --accent: #2f6fb0; --warn: #b3541e;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font-family: "Segoe UI", Roboto, Arial, sans-serif; line-height: 1.5; }
header { background: linear-gradient(135deg, #1f3f66, #2f6fb0); color: #fff; padding: 28px 32px 22px; }
header h1 { margin: 0 0 6px; font-size: 27px; font-weight: 600; }
header p { margin: 2px 0; opacity: .9; font-size: 14px; }
main { max-width: 1180px; margin: 0 auto; padding: 22px 20px 60px; }
section { margin-bottom: 30px; }
h2 { font-size: 20px; margin: 26px 0 12px; padding-bottom: 6px; border-bottom: 2px solid var(--line); }
h3 { font-size: 15px; margin: 18px 0 8px; color: var(--muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: .04em; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr)); gap: 12px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.card .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }
.card .value { font-size: 23px; font-weight: 600; margin-top: 4px; }
.card .hint { font-size: 12px; color: var(--muted); margin-top: 2px; }
.map-wrap { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
#map { height: 620px; border-radius: 8px; background: #eef1f5; }
#map-fallback { display: none; padding: 30px; text-align: center; color: var(--muted); }
.map-controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 10px; }
select, button { font: inherit; padding: 7px 10px; border-radius: 7px; border: 1px solid var(--line);
  background: #fff; color: var(--ink); }
select { min-width: 320px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; align-items: center; }
.legend .item { display: flex; align-items: center; gap: 5px; font-size: 12px; color: var(--muted); }
.legend .swatch { width: 22px; height: 12px; border-radius: 2px; border: 1px solid rgba(0,0,0,.12); }
.desc { font-size: 13px; color: var(--muted); margin: 8px 0 0; }
#block-info { margin-top: 10px; font-size: 13px; background: #f8fafc; border: 1px dashed var(--line);
  border-radius: 8px; padding: 10px 12px; min-height: 20px; color: var(--muted); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap: 16px; }
.chart-card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; }
svg.chart { width: 100%; height: auto; display: block; }
.chart-title { font-size: 13px; font-weight: 600; fill: #1c2430; }
.chart-note { font-size: 11px; fill: #8a96a3; }
.chart-empty { font-size: 13px; fill: #8a96a3; }
.axis { font-size: 11px; fill: #6b7787; }
.axis-title { font-size: 11px; fill: #6b7787; }
.grid { stroke: #eef1f5; stroke-width: 1; }
.axis-line { stroke: #cfd6df; stroke-width: 1; }
table.tbl { border-collapse: collapse; width: 100%; background: var(--card); font-size: 13px;
  border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
table.tbl th { background: #eef2f7; text-align: left; padding: 8px 10px; font-weight: 600;
  border-bottom: 1px solid var(--line); white-space: nowrap; }
table.tbl td { padding: 7px 10px; border-bottom: 1px solid #f0f3f7; }
table.tbl tr:last-child td { border-bottom: none; }
table.tbl tr:hover td { background: #f8fbff; }
.table-scroll { overflow-x: auto; }
.notes { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 16px 20px; }
.notes ul { margin: 6px 0 0 18px; padding: 0; }
.notes li { margin-bottom: 6px; font-size: 14px; }
.warn { color: var(--warn); }
.leaflet-tooltip.tip { font-size: 12px; line-height: 1.45; max-width: 260px; }
footer { text-align: center; color: var(--muted); font-size: 12px; padding: 20px; }
@media (max-width: 720px) { .charts { grid-template-columns: 1fr; } #map { height: 460px; } }
"""

MAP_JS = """
(function () {
  var mapEl = document.getElementById('map');
  if (!window.L) {
    mapEl.style.display = 'none';
    document.getElementById('map-fallback').style.display = 'block';
    return;
  }

  var current = METRICS[0].key;
  var map = L.map('map', {zoomControl: true, preferCanvas: true}).setView([CENTER_LAT, CENTER_LON], 11);
  var tiles = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    maxZoom: 19,
    attribution: 'Кварталы: BlocksNet / OpenStreetMap. Подложка: © OpenStreetMap, © CARTO'
  }).addTo(map);

  function styleFor(feature) {
    return {
      color: '#5b6470', weight: 0.45, opacity: 0.8,
      fillColor: feature.properties['c_' + current] || NO_DATA,
      fillOpacity: 0.82
    };
  }

  var layer = L.geoJSON(DATA, {
    style: styleFor,
    onEachFeature: function (feature, lyr) {
      lyr.bindTooltip(feature.properties.tip, {sticky: true, className: 'tip', direction: 'top'});
      lyr.on('mouseover', function () { lyr.setStyle({weight: 1.6, color: '#1c2430'}); });
      lyr.on('mouseout', function () { lyr.setStyle({weight: 0.45, color: '#5b6470'}); });
      lyr.on('click', function () {
        var info = document.getElementById('block-info');
        info.innerHTML = feature.properties.tip;
      });
    }
  }).addTo(map);

  map.fitBounds(layer.getBounds(), {padding: [12, 12]});

  var select = document.getElementById('metric');
  METRICS.forEach(function (m, i) {
    var opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = m.title;
    if (i === 0) { opt.selected = true; }
    select.appendChild(opt);
  });

  function renderLegend(metric) {
    var box = document.getElementById('legend');
    box.innerHTML = '';
    metric.legend.forEach(function (item) {
      var wrap = document.createElement('span');
      wrap.className = 'item';
      var sw = document.createElement('span');
      sw.className = 'swatch';
      sw.style.background = item.color;
      var txt = document.createElement('span');
      txt.textContent = item.label;
      wrap.appendChild(sw);
      wrap.appendChild(txt);
      box.appendChild(wrap);
    });
    document.getElementById('metric-desc').textContent = metric.desc;
  }

  function applyMetric(key) {
    current = key;
    layer.eachLayer(function (lyr) {
      lyr.setStyle({fillColor: lyr.feature.properties['c_' + key] || NO_DATA});
    });
    var metric = METRICS.filter(function (m) { return m.key === key; })[0];
    if (metric) { renderLegend(metric); }
  }

  select.addEventListener('change', function (e) { applyMetric(e.target.value); });
  renderLegend(METRICS[0]);

  document.getElementById('basemap').addEventListener('change', function (e) {
    if (e.target.checked) { tiles.addTo(map); } else { map.removeLayer(tiles); }
  });

  document.getElementById('reset-view').addEventListener('click', function () {
    map.fitBounds(layer.getBounds(), {padding: [12, 12]});
  });
})();
"""


def build_html(key, meta, data, stats, specs, features, charts, tables, conclusions, static_maps):
    blocks = data["blocks"]
    center_lat = float(blocks.to_crs("EPSG:4326").total_bounds[1] + blocks.to_crs("EPSG:4326").total_bounds[3]) / 2
    center_lon = float(blocks.to_crs("EPSG:4326").total_bounds[0] + blocks.to_crs("EPSG:4326").total_bounds[2]) / 2

    cards = [
        ("Территория", stats["territory_name"], ""),
        ("Кварталов", num(stats["n_blocks"]), ""),
        ("Площадь", num(stats["area_km2"], 1) + " км²", ""),
        ("Население", num(stats["population"]), "модельное"),
        ("Школ", num(stats["schools"]), num(stats["seats"]) + " мест"),
        ("Медиана FSI", num(stats["median_fsi"], 2), "застроенные кварталы"),
        ("Медиана доступности", num(stats["median_acc"], 1) + " мин", "недостижимое = 60"),
        ("Плотность", num(stats["density"]) + " чел/км²", ""),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>'
        f'<div class="hint">{escape(hint)}</div></div>'
        for label, value, hint in cards
    )

    metrics_json = json.dumps(map_metrics_json(specs), ensure_ascii=False)
    data_json = json.dumps(features, ensure_ascii=False, separators=(",", ":"))

    charts_html = "".join(f'<div class="chart-card">{svg}</div>' for svg in charts)

    tables_html = []
    for title, table in tables:
        if not table:
            continue
        tables_html.append(f"<h3>{escape(title)}</h3><div class='table-scroll'>{table}</div>")
    tables_html = "\n".join(tables_html)

    static_html = ""
    if static_maps:
        items = "".join(
            f'<figure><img src="data:image/png;base64,{encoded}" alt="{escape(title)}">'
            f'<figcaption>{escape(title)}</figcaption></figure>'
            for title, encoded in static_maps
        )
        static_html = (
            "<details><summary>Статичные карты из ноутбука (PNG)</summary>"
            f"<div class='static-maps'>{items}</div></details>"
        )

    page = Template(PAGE_TEMPLATE).substitute(
        style=STYLE.replace("$", "$$"),
        title=escape("Ставрополь — учебное исследование территории"),
        territory=escape(stats["territory_name"]),
        built_at=datetime.now().strftime("%d.%m.%Y %H:%M"),
        cards=cards_html,
        summary=summary_table(stats),
        charts=charts_html,
        tables=tables_html,
        static_maps=static_html,
        conclusions=conclusions,
        data_json=data_json,
        metrics_json=metrics_json,
        center_lat=f"{center_lat:.5f}",
        center_lon=f"{center_lon:.5f}",
        map_js=MAP_JS.replace("$", "$$"),
        no_data=NO_DATA_COLOR,
        key=escape(key),
    )
    return page


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>$style</style>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>
<header>
  <h1>$title</h1>
  <p>Территория: $territory &nbsp;·&nbsp; файл: dashboard.html &nbsp;·&nbsp; собран $built_at</p>
  <p>Данные: OpenStreetMap · инструменты: BlocksNet 1.0.0a9, OSMnx, GeoPandas · пешеходный граф, 5 км/ч</p>
</header>
<main>

<section>
  <div class="cards">$cards</div>
</section>

<section>
  <h2>Карта кварталов</h2>
  <div class="map-wrap">
    <div class="map-controls">
      <label for="metric">Показатель на карте:</label>
      <select id="metric"></select>
      <label><input type="checkbox" id="basemap" checked> подложка карты</label>
      <button id="reset-view" type="button">весь город</button>
    </div>
    <div id="map"></div>
    <div id="map-fallback">
      Карта не загрузилась: для интерактивной карты нужен интернет (библиотека Leaflet и подложка).
      Таблицы, графики и выводы ниже работают и без интернета.
    </div>
    <div class="legend" id="legend"></div>
    <p class="desc" id="metric-desc"></p>
    <div id="block-info">Нажмите на квартал — здесь появятся его показатели.</div>
  </div>
</section>

<section>
  <h2>Сводка показателей</h2>
  <div class="table-scroll">$summary</div>
</section>

<section>
  <h2>Графики</h2>
  <div class="charts">$charts</div>
</section>

<section>
  <h2>Таблицы</h2>
  $tables
  $static_maps
</section>

<section>
  <h2>Выводы</h2>
  <div class="notes">$conclusions</div>
</section>

<section>
  <h2>Как это считалось и чего в данных нет</h2>
  <div class="notes">
    <ul>
      <li><b>Кварталы</b> — участки, ограниченные дорогами, водой и железными дорогами
        (нарезка BlocksNet). Площадь территории — сумма площадей кварталов.</li>
      <li><b>Население</b> — модельное: жилая площадь зданий делится на 30 м² на человека.
        Это не перепись: официальное население Ставрополя может отличаться.</li>
      <li><b>Этажность</b> берётся из тега <code>building:levels</code>. Где тега нет, принят
        1 этаж, поэтому FSI и L скорее занижены.</li>
      <li><b>Школы</b> — из OpenStreetMap (<code>amenity=school</code>). Вместимость берётся из OSM,
        при пропуске — 250 мест. Норматив — 120 мест на 1000 жителей, радиус доступности — 15 минут пешком.</li>
      <li><b>Обеспеченность</b> считается с конкуренцией: места школы достаются ближайшим кварталам,
        поэтому при формально достаточном числе мест часть кварталов может остаться с нулём.</li>
      <li><b>Доступность территории</b> — средневзвешенное время до всех кварталов города
        (вес — площадь квартала). Недостижимые пары в варианте «cap 60» заменены на 60 минут.</li>
      <li><b>Морфотипы</b> Strelka и Spacematrix — два разных взгляда на тип застройки;
        у кварталов без зданий тип не определяется.</li>
      <li><b>Граф</b> — пешеходный, OpenStreetMap, скорость 5 км/ч. Время в минутах.</li>
    </ul>
  </div>
</section>

</main>
<footer>
  Файл собран скриптом make_dashboard.py из готовых данных папки data_stavropol.
</footer>

<script>
const NO_DATA = "$no_data";
const DATA = $data_json;
const METRICS = $metrics_json;
const CENTER_LAT = $center_lat;
const CENTER_LON = $center_lon;
$map_js
</script>
</body>
</html>
"""


def build_charts(blocks, stats):
    charts = []

    built = blocks[blocks["fsi"].fillna(0) > 0]
    charts.append(svg_histogram(
        built["fsi"], "Распределение FSI (застроенные кварталы)",
        "FSI, м² этажей на м² квартала", bins=22, fmt="{:.2f}",
        note=f"кварталов: {num(len(built))}",
    ))
    charts.append(svg_histogram(
        built["l"], "Распределение этажности L",
        "этажей", bins=18, color="#6a9fb5", fmt="{:.1f}",
        note="где этажность не указана в OSM — принят 1 этаж",
    ))
    charts.append(svg_histogram(
        blocks["area_accessibility_cap60"], "Доступность территории",
        "минут", bins=20, color="#8f6bb0", fmt="{:.0f}",
        note="недостижимые пары заменены на 60 минут",
    ))
    if "building_count" in built.columns:
        charts.append(svg_histogram(
            built["building_count"], "Распределение числа зданий", "зданий в квартале",
            bins=20, color="#4e79a7", fmt="{:.0f}",
            note=f"застроенных кварталов: {num(len(built))}",
        ))
    if "demand" in blocks.columns:
        need = blocks[blocks["demand"].fillna(0) > 0]
        if not need.empty:
            charts.append(svg_histogram(
                need["provision_weak"], "Обеспеченность всеми местами", "доля спроса",
                bins=20, color="#1a9850", fmt="{:.2f}",
                note="включая далёкие школы, а не только 15 минут",
            ))

    land_use = blocks.dropna(subset=["land_use"])
    if not land_use.empty:
        area_by_use = (land_use.groupby("land_use")["site_area"].sum() / 1e6).sort_values(ascending=False)
        items = []
        for name, value in area_by_use.head(8).items():
            items.append((LAND_USE_RU.get(name, name), float(value), None))
        charts.append(svg_bars(items, "Площадь по землепользованию", fmt=lambda v: num(v, 1, " км²")))

        pop_by_use = land_use.groupby("land_use")["population"].sum().sort_values(ascending=False)
        items = [(LAND_USE_RU.get(name, name), float(value), None) for name, value in pop_by_use.head(8).items()]
        charts.append(svg_bars(items, "Население по землепользованию", fmt=lambda v: num(v)))

    morph = blocks["strelka_morphotype"].dropna()
    if not morph.empty:
        counts = morph.value_counts().head(8)
        items = [(ru_morphotype(name), int(value), None) for name, value in counts.items()]
        charts.append(svg_bars(items, "Морфотипы Strelka", fmt=lambda v: num(v)))

    if "demand" in blocks.columns:
        demand_blocks = blocks[blocks["demand"].fillna(0) > 0]
        if not demand_blocks.empty:
            values = demand_blocks["provision_strong"].fillna(0)
            labels = ["0%", "до 25%", "25–50%", "50–75%", "75–100%", ">100%"]
            counts = pd.cut(
                values, bins=[-1e-9, 1e-9, 0.25, 0.5, 0.75, 1.0, np.inf], labels=labels
            ).value_counts().reindex(labels).fillna(0)
            colors = ["#d73027", "#f46d43", "#fdae61", "#fee090", "#a6d96a", "#1a9850"]
            items = [(label, int(count), colors[i]) for i, (label, count) in enumerate(counts.items())]
            charts.append(svg_bars(
                items, "Обеспеченность школами (кварталы с потребностью)",
                fmt=lambda v: num(v), note=f"всего таких кварталов: {num(len(demand_blocks))}",
            ))
    return charts


def provision_links_table(links):
    if links is None or links.empty or "value" not in links.columns:
        return None
    frame = links.copy()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    source = "source" if "source" in frame.columns else frame.columns[0]
    grouped = (
        frame.groupby(source)["value"].sum()
        .sort_values(ascending=False).head(10).reset_index()
    )
    grouped.columns = ["Квартал-источник", "Распределено учеников"]
    grouped["Распределено учеников"] = grouped["Распределено учеников"].map(lambda v: num(v))
    return df_to_table(grouped)


def build_tables(data, blocks, stats):
    tables = []
    tables.append(("Все показатели: распределение по кварталам", indicator_profile_table(blocks)))
    tables.append(("Землепользование", land_use_table(blocks)[0]))
    tables.append(("Морфотипы Strelka", morphotype_table(blocks, "strelka_morphotype", "Морфотип")))
    tables.append(("Морфотипы Spacematrix", morphotype_table(blocks, "spacematrix_morphotype", "Морфотип")))
    tables.append(("Кластеры Spacematrix", spacematrix_table(data["spacematrix_clusters"], blocks)))

    provision, worst = provision_table(blocks)
    tables.append(("Обеспеченность школами: распределение кварталов", provision))
    tables.append(("Топ-10 кварталов с самой низкой обеспеченностью", worst))
    tables.append(("Связи «квартал → школа»: топ-10 по числу учеников",
                   provision_links_table(data["provision_links"])))

    tables.append(("Топ-10 кварталов по населению",
                   top_blocks_table(blocks, "population", "Население", digits=0)))
    tables.append(("Топ-10 кварталов по FSI",
                   top_blocks_table(blocks, "fsi", "FSI", digits=2)))
    return tables


def find_static_maps(key):
    """Берёт готовые PNG из папки results и встраивает их в страницу."""
    wanted = [
        ("Население кварталов", f"{key}_final_population.png"),
        ("FSI", f"{key}_final_fsi.png"),
        ("Морфотипы застройки", f"{key}_final_morphotypes.png"),
        ("Обеспеченность школами", f"{key}_final_provision.png"),
    ]
    maps = []
    for title, filename in wanted:
        path = RESULTS_DIR / filename
        if path.exists():
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            maps.append((title, encoded))
    return maps


def main():
    print("Собираю дашборд из готовых файлов...")

    key, meta = find_territory_key()
    print(f"   территория: {key}")

    data = load_inputs(key)
    blocks = data["blocks"]
    if "block_id" not in blocks.columns:
        blocks = blocks.reset_index().rename(columns={"index": "block_id"})
        data["blocks"] = blocks

    stats = compute_stats(blocks, meta)
    specs = build_metric_specs(blocks)
    map_data = build_map_payload(blocks, specs)
    charts = build_charts(blocks, stats)
    tables = build_tables(data, blocks, stats)
    conclusions = build_conclusions(stats, blocks)
    static_maps = find_static_maps(key)

    page = build_html(key, meta, data, stats, specs, map_data, charts, tables, conclusions, static_maps)
    OUT_PATH.write_text(page, encoding="utf-8")

    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"   кварталов на карте: {len(map_data['features'])}")
    print(f"   слоёв карты: {len(specs)}, графиков: {len(charts)}, таблиц: {len([t for _, t in tables if t])}")
    print()
    print(f"✅ Готово: {OUT_PATH}")
    print(f"   размер файла: {size_mb:.1f} МБ")
    print("   откройте двойным щелчком или командой:")
    print(f'   start "" "{OUT_PATH}"')


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print()
        print("❌ Не удалось собрать дашборд:", type(exc).__name__, exc)
        print("   Проверьте, что ноутбук выполнен до раздела 12 и папка data_stavropol/results заполнена.")
        sys.exit(1)
