"""Build the three GeoJSON overlays used by the map page (data/geo/*.geojson).

    python scripts/build_geojson.py

Sources / assumptions
---------------------
* EL   - price areas NO1..NO5. Taken from the electricitymaps-contrib
         ``geo/world.geojson`` (zones NO-NO1 .. NO-NO5). Coarse (simplified)
         outlines: fine for a choropleth overview, not for cartography.
* NO   - Norway as a whole = union of the five price areas (area_nr 0).
* VASS - NVE "vassdragsomraade" 1..3. NVE does NOT publish the boundaries
         together with Magasinstatistikk, so this file is a PLACEHOLDER made of
         three latitude bands. Replace data/geo/VASS.geojson with the real
         polygons (keep the property ``area_nr`` = 1, 2, 3 and ``name``);
         the app reads the file as is and shows a warning while the
         placeholder flag (``"placeholder": true``) is present.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

SRC = ("https://raw.githubusercontent.com/electricitymaps/electricitymaps-contrib/"
       "master/geo/world.geojson")
OUT = Path(__file__).resolve().parent.parent / "data" / "geo"

EL_NAMES = {
    1: "NO1 - Eastern Norway",
    2: "NO2 - South-western Norway",
    3: "NO3 - Central Norway",
    4: "NO4 - Northern Norway",
    5: "NO5 - Western Norway",
}


def _round(geom, nd=3):
    def r(c):
        return [round(x, nd) for x in c] if isinstance(c[0], (int, float)) else [r(i) for i in c]
    m = mapping(geom)
    return {"type": m["type"], "coordinates": r(m["coordinates"])}


def _fc(features, **extra):
    return {"type": "FeatureCollection", **extra, "features": features}


def _feature(nr, area_type, name, geom, **props):
    return {"type": "Feature",
            "properties": {"area_type": area_type, "area_nr": nr, "name": name, **props},
            "geometry": _round(geom)}


def main() -> None:
    with urllib.request.urlopen(SRC, timeout=120) as r:  # noqa: S310 (fixed https URL)
        world = json.load(r)

    el = {}
    for f in world["features"]:
        z = f["properties"].get("zoneName", "")
        if z.startswith("NO-NO") and z[-1].isdigit():
            el[int(z[-1])] = shape(f["geometry"]).buffer(0)
    if sorted(el) != [1, 2, 3, 4, 5]:
        sys.exit(f"Expected NO1..NO5, found {sorted(el)}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "EL.geojson").write_text(json.dumps(_fc(
        [_feature(n, "EL", EL_NAMES[n], g) for n, g in sorted(el.items())])))

    norway = unary_union(list(el.values()))
    (OUT / "NO.geojson").write_text(json.dumps(_fc(
        [_feature(0, "NO", "Norway", norway)])))

    bands = {1: (57, 62), 2: (62, 66), 3: (66, 72)}   # placeholder, see docstring
    feats = []
    for n, (lo, hi) in bands.items():
        part = norway.intersection(box(0, lo, 35, hi))
        feats.append(_feature(n, "VASS", f"Water area {n} (placeholder)", part, placeholder=True))
    (OUT / "VASS.geojson").write_text(json.dumps(_fc(feats, placeholder=True)))

    for p in sorted(OUT.glob("*.geojson")):
        print(p.name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
