"""Build ffw/data/worlds.json from the Fifth Frontier War map.

The world statistics below were transcribed by hand from the 146 world-surface
boxes printed around the edge of the Fifth Frontier War game map (pages 27-30 of
the scanned rules PDF).  Each box gives, in order:

    starport type,  base symbols,  tech level
    SDB factor,     water dot,     defence battalion factor
    world name

Box background colour encodes atmosphere (blue = breathable, grey = tainted or
hostile, white = vacuum) and the surrounding frame encodes travel zone
(yellow = amber, red = red).

Hex locations, gas giants and political allegiance come from the canonical
Traveller Spinward Marches sector data (tools/spin.tsv, downloaded from
travellermap.com).  Fifth Frontier War uses map-local hex numbers that are the
canonical sector hex plus four columns and four rows -- e.g. Regina is canonical
1910 and FFW 2314.  The three Vargr worlds and Pandrin lie coreward of the
sector boundary and are entered by hand from the stellar display.

Running this script regenerates ffw/data/worlds.json; the generated file is
committed so the game works offline.
"""

from __future__ import annotations

import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "ffw", "data", "worlds.json")

BREATH, TAINT, VAC = "breathable", "tainted", "vacuum"

# name, starport, tech level, SDB factor, defence battalions, water, atmosphere,
# bases (N naval, S scout, M non-Imperial military), travel zone
WORLD_BOXES = [
    # ---- map page 27 (coreward-spinward quadrant) ----------------------------
    ("Algebaster",   "C",  9,    0,     15, True,  BREATH, "",   None),
    ("Esalin",       "C",  8,    0,      0, True,  BREATH, "",   "amber"),
    ("Jewell",       "A", 12,  120,   1200, True,  TAINT,  "SN", None),
    ("Mongo",        "A", 10,    0,     15, True,  BREATH, "SN", None),
    ("Nerewhon",     "E",  7,    0,      0, True,  TAINT,  "",   None),
    ("Plaven",       "E",  3,    0,      0, True,  TAINT,  "",   None),
    ("Riverland",    "C",  9, 1000, 150000, True,  BREATH, "",   "amber"),
    ("Stave",        "E",  2,    0,      0, True,  BREATH, "",   None),
    ("Whenge",       "D",  8,    0,      1, True,  TAINT,  "",   None),
    ("Zircon",       "C",  9,    0,      1, True,  TAINT,  "S",  None),
    ("Ao-dai",       "E",  6,    0,      3, False, VAC,    "",   None),
    ("Farreach",     "A", 11,    0,      0, False, VAC,    "M",  "amber"),
    ("Louzy",        "D",  8,  500,  20000, True,  TAINT,  "",   None),
    ("Nakege",       "D",  2,    0,      0, True,  TAINT,  "",   "amber"),
    ("Ninjar",       "A", 12,    0,      1, True,  VAC,    "M",  None),
    ("Quar",         "B", 11,    1,     12, True,  TAINT,  "N",  "amber"),
    ("Ruby",         "B", 11,    0,      0, False, VAC,    "S",  None),
    ("Utoland",      "C",  7,    0,      0, True,  TAINT,  "",   None),
    ("Zenopit",      "D",  7,    0,      1, False, TAINT,  "",   None),
    ("871-438",      "E",  0,    0,      0, False, VAC,    "",   None),
    ("Atsa",         "B", 10,    1,     15, True,  TAINT,  "M",  None),
    ("Foelen",       "B",  8,    0,      2, True,  TAINT,  "",   "amber"),
    ("Lysen",        "B", 10,    0,      1, True,  TAINT,  "S",  None),
    ("Narval",       "D",  6,    0,      3, True,  TAINT,  "M",  "amber"),
    ("Pequan",       "E",  4,    0,      0, True,  BREATH, "",   None),
    ("Rasatt",       "E",  7,    0,      1, True,  BREATH, "",   None),
    ("Sheyou",       "B", 10,    1,    150, True,  BREATH, "M",  None),
    ("Chwistyoch",   "B", 10,    1,    150, True,  BREATH, "M",  None),
    ("Frond",        "E",  9,    0,      0, False, TAINT,  "",   None),
    ("Cipango",      "A", 12,   12,   1200, True,  BREATH, "M",  None),
    ("Gesentown",    "B", 12,    0,      1, True,  VAC,    "M",  None),
    ("Clan",         "B", 10,   10,    150, True,  TAINT,  "M",  "amber"),
    ("Gougeste",     "C", 10,    0,      0, True,  TAINT,  "",   "amber"),
    ("Chronor",      "A", 13,  150,   1000, True,  TAINT,  "M",  None),
    ("Grant",        "X",  0,    0,      0, True,  BREATH, "",   "red"),
    ("Emerald",      "B", 11,    0,      1, True,  BREATH, "S",  None),
    ("Indo",         "E",  5,    0,      0, True,  TAINT,  "",   None),
    # ---- map page 28 (coreward-trailing quadrant) ----------------------------
    ("Balent",       "C",  5,    0,      0, True,  TAINT,  "M",  None),
    ("Pandrin",      "B",  3,    0,      0, False, BREATH, "",   None),
    ("Gireel",       "B",  9,    0,      0, True,  TAINT,  "M",  None),
    ("Uthilh",       "A", 10,    0,      0, True,  BREATH, "M",  None),
    ("Alell",        "B", 10,   10,   1500, True,  BREATH, "",   None),
    ("Beck's World", "D",  4,    0,      0, True,  BREATH, "",   None),
    ("Dentus",       "C", 10,    0,      0, True,  TAINT,  "S",  None),
    ("Heya",         "B",  5,    0,      0, True,  BREATH, "",   None),
    ("Paya",         "A",  9,    0,      0, True,  BREATH, "N",  None),
    ("Uakye",        "B", 13,    0,      0, True,  TAINT,  "",   None),
    ("Algine",       "X",  4,    0,      0, True,  BREATH, "",   "red"),
    ("Boughene",     "A", 13,    0,      0, False, TAINT,  "S",  None),
    ("Dhian",        "C",  4,    0,      0, False, TAINT,  "",   None),
    ("Inthe",        "B",  9,    1,     15, True,  TAINT,  "SN", None),
    ("Pixie",        "A", 13,    0,      0, False, VAC,    "N",  None),
    ("Violante",     "C", 10,    0,      0, True,  BREATH, "",   None),
    ("Efate",        "A", 13,  150,   1000, True,  TAINT,  "N",  None),
    ("Jenghe",       "C",  9,    0,      1, True,  TAINT,  "S",  None),
    ("Pscias",       "X",  1,    0,      0, True,  BREATH, "",   "red"),
    ("Whanga",       "E",  7,    0,      0, True,  TAINT,  "",   None),
    ("Enope",        "C",  6,    0,   3000, True,  VAC,    "",   None),
    ("Keng",         "E",  3,    0,      0, True,  TAINT,  "",   None),
    ("Regina",       "A", 10,   10,   1500, True,  BREATH, "SN", None),
    ("Wochiers",     "E",  9,   10,    150, False, TAINT,  "",   None),
    ("Feri",         "B", 11,   12,   1200, True,  BREATH, "S",  None),
    ("Kinorb (2202)","A",  5,    0,      0, True,  BREATH, "",   None),
    ("Rethe",        "E",  8,  500,  20000, False, TAINT,  "",   None),
    ("Yorbund",      "C",  7,    0,      1, False, TAINT,  "",   None),
    ("Focaline",     "E", 10,    0,      1, True,  BREATH, "",   None),
    ("Knorbes",      "E",  2,    0,      0, True,  BREATH, "",   None),
    ("Roup",         "C",  6,    0,   3000, True,  TAINT,  "S",  "amber"),
    ("Yori",         "C", 13,    1,    100, False, BREATH, "",   None),
    ("Forboldn",     "E",  4,    0,      0, True,  TAINT,  "",   None),
    ("Menorb",       "C",  7,   50,  20000, True,  BREATH, "",   None),
    ("Ruie",         "C",  7,   50,   2000, True,  TAINT,  "",   "amber"),
    ("Yres",         "B",  7,    0,     20, False, VAC,    "",   None),
    ("Hefry",        "C",  7,    0,      0, False, VAC,    "S",  None),
    ("Moughas",      "C", 11,    0,      1, True,  BREATH, "",   None),
    ("Shionthy",     "X",  8,    0,     20, False, VAC,    "",   "red"),
    ("Yurst",        "E",  5,    0,      0, False, VAC,    "",   None),
    # ---- map page 29 (rimward-spinward quadrant) -----------------------------
    ("Arden",        "C",  8,   50,  20000, True,  BREATH, "",   None),
    ("Denotam",      "B", 10,    0,      0, True,  TAINT,  "N",  None),
    ("Gram",         "A", 10,  120,   1200, True,  TAINT,  "M",  None),
    ("Mjolnir",      "B", 10,    0,      0, False, TAINT,  "M",  None),
    ("Arkadia",      "E",  6,    0,    300, True,  TAINT,  "",   None),
    ("Digitis",      "E",  5,    0,      0, True,  TAINT,  "",   None),
    ("Gungnir",      "B",  8,    0,     20, True,  TAINT,  "M",  None),
    ("Phlume",       "C",  8,    0,     20, True,  BREATH, "",   None),
    ("Sansibar",     "B", 10,    0,      0, False, VAC,    "M",  None),
    ("Tionale",      "C",  8,    0,      0, True,  TAINT,  "",   "amber"),
    ("Asgard",       "X",  2,    0,      0, True,  TAINT,  "",   "red"),
    ("Edinina",      "E",  5,    0,      0, False, VAC,    "",   "amber"),
    ("Joyeuse",      "B", 10,    1,    150, True,  BREATH, "M",  "amber"),
    ("Quare",        "B",  9,    0,      0, False, VAC,    "",   None),
    ("Saurus",       "D",  7,    0,      2, True,  BREATH, "",   None),
    ("Tremous Dex",  "B", 12,    0,      0, True,  VAC,    "",   None),
    ("Asmodeus",     "E",  4,    0,      0, True,  TAINT,  "",   None),
    ("Ficant",       "E",  5,    0,      0, True,  BREATH, "",   None),
    ("Lebeau",       "B", 12,    0,      1, True,  BREATH, "",   None),
    ("Querion",      "B",  9,    1,    150, True,  BREATH, "M",  None),
    ("Stellatio",    "D",  4,    0,      0, False, TAINT,  "",   None),
    ("Vilis",        "A", 10,  100,   1500, True,  TAINT,  "",   None),
    ("Calit",        "C",  7,    5,    200, True,  TAINT,  "",   None),
    ("Frenzie",      "A", 10,    0,      0, False, VAC,    "N",  None),
    ("Mirriam",      "E",  8,    0,      0, True,  TAINT,  "N",  None),
    ("Rangent",      "E",  7,    0,      2, True,  TAINT,  "",   None),
    ("Tavonni",      "E",  0,    0,      0, True,  BREATH, "",   None),
    ("728-907",      "D",  0,    0,      0, True,  BREATH, "",   None),
    ("Caloran",      "D",  5,    0,      0, True,  TAINT,  "",   None),
    ("Garda-Vilis",  "B", 10,   10,    150, True,  TAINT,  "S",  None),
    ("Mizan-fel",    "B",  8,    0,      2, True,  BREATH, "",   None),
    ("Rapp's World", "C",  8,    0,      0, True,  TAINT,  "M",  None),
    ("Terra Nova",   "C", 10,    0,      0, True,  BREATH, "",   None),
    ("899-076",      "E",  8,    0,      0, True,  VAC,    "",   None),
    # ---- map page 30 (rimward-trailing quadrant) -----------------------------
    ("Pirema",       "D",  5,    0,      0, True,  TAINT,  "",   None),
    ("Treece",       "D",  8,    5,    200, True,  TAINT,  "",   None),
    ("Porozlo",      "A", 10, 1000, 150000, True,  BREATH, "",   None),
    ("Tureded",      "C",  9,    0,      1, True,  BREATH, "",   None),
    ("Fulacin",      "A", 13,    0,      0, True,  TAINT,  "",   None),
    ("Keanou",       "C",  7,    0,      0, False, TAINT,  "S",  None),
    ("Quopist",      "B", 10,    0,     15, True,  BREATH, "",   "amber"),
    ("Valhalla",     "E",  5,    0,      0, True,  BREATH, "",   None),
    ("Ghandi",       "B", 10,    0,      0, True,  VAC,    "N",  None),
    ("K'Kirka",      "C",  8,    0,      0, False, TAINT,  "",   None),
    ("Rech",         "D",  6,    0,      3, True,  TAINT,  "",   None),
    ("Victoria",     "X",  4,    0,      0, True,  VAC,    "",   "red"),
    ("Djinni",       "E",  0,    0,      0, True,  BREATH, "",   "red"),
    ("Gileden",      "C",  5,    0,      0, True,  BREATH, "",   None),
    ("Kinorb (2512)","C",  9,    0,      0, True,  TAINT,  "",   None),
    ("Rhise",        "C", 10,    0,      0, False, VAC,    "",   "amber"),
    ("Vreibefger",   "E",  2,    0,      0, True,  BREATH, "",   None),
    ("Dinom",        "D", 10,    0,      0, False, VAC,    "",   None),
    ("Echiste",      "C", 10,    0,      0, True,  TAINT,  "",   None),
    ("Icetina",      "B",  7,    0,      1, True,  TAINT,  "N",  None),
    ("La'Belle",     "C",  3,    0,      0, True,  BREATH, "",   None),
    ("Rhylanor",     "A", 15,  200,    500, True,  TAINT,  "SN", None),
    ("Wypoc",        "C", 12,    0,      0, False, TAINT,  "",   "amber"),
    ("Dinomn",       "B",  3,    0,      0, True,  TAINT,  "S",  None),
    ("Equus",        "B", 11,   12,   1200, True,  BREATH, "S",  None),
    ("Ivendo",       "B", 10,    0,      1, True,  TAINT,  "SN", None),
    ("Lanth",        "A", 11,    0,      0, True,  TAINT,  "SN", None),
    ("Risek",        "A", 10,    0,      0, True,  TAINT,  "N",  None),
    ("Ylaven",       "X",  4,    0,      0, True,  BREATH, "",   "red"),
    ("D'Ganzio",     "B", 13,    0,      0, True,  TAINT,  "N",  None),
    ("Extolay",      "B", 10,   10,   1500, True,  BREATH, "",   None),
    ("Jae Tellona",  "A",  8,    0,      2, False, BREATH, "N",  None),
    ("Macene",       "B", 14,    0,      0, False, VAC,    "N",  None),
    ("Sonthert",     "X",  3,    0,      0, True,  TAINT,  "",   "red"),
    ("Zivije",       "C", 11,  120,   1200, False, TAINT,  "",   None),
]

# Worlds coreward of the Spinward Marches sector boundary, read straight off the
# stellar display (hex, gas giant).  The Vargr border on the map encloses only
# Gireel, Uthilh and Balent -- consistent with the 3-Vargr-world victory bonus.
OFF_SECTOR = {
    "Gireel":  ("2702", True,  "vargr"),
    "Uthilh":  ("2703", True,  "vargr"),
    "Balent":  ("2704", True,  "vargr"),
    "Pandrin": ("2604", False, "neutral"),
}

# Fifth Frontier War hex is the canonical Spinward Marches hex offset by this.
HEX_OFFSET = (4, 4)

ALLEGIANCE = {
    "ImDd": "imperial",
    "ZhIN": "zhodani",
    "SwCf": "sword_worlds",
}

# Rule 5, Special Situations: Quar and Zircon lie outside the Imperium but hold
# Imperial bases and count as Imperial-controlled.
FORCE_IMPERIAL = {"Quar", "Zircon"}

SUBSECTOR_CAPITALS = {
    "0708", "1018", "1627", "1510", "1520", "2123", "2314", "3120",
}


def shift(hex_str: str) -> str:
    col, row = int(hex_str[:2]), int(hex_str[2:])
    return "%02d%02d" % (col + HEX_OFFSET[0], row + HEX_OFFSET[1])


def load_canonical():
    by_name = {}
    with open(os.path.join(HERE, "spin.tsv"), newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            by_name.setdefault(row["Name"].strip().lower(), []).append(row)
    return by_name


def main() -> None:
    canon = load_canonical()
    worlds = []
    used_hexes = {}
    for (name, port, tl, sdb, defbn, water, atm, bases, zone) in WORLD_BOXES:
        key = name.lower()
        lookup = key
        gg = False
        if name in OFF_SECTOR:
            hexid, gg, owner = OFF_SECTOR[name]
        else:
            want_hex = None
            m = re.search(r"\((\d{4})\)$", name)
            if m:
                lookup, want_hex = key.split(" (")[0], m.group(1)
            rows = canon.get(lookup)
            if not rows:
                raise SystemExit("no canonical entry for %r" % name)
            if want_hex:
                row = [r for r in rows if r["Hex"] == want_hex][0]
            elif len(rows) > 1:
                inside = [r for r in rows
                          if 3 <= int(r["Hex"][:2]) <= 27 and 1 <= int(r["Hex"][2:]) <= 23]
                if len(inside) != 1:
                    raise SystemExit("ambiguous canonical entry for %r" % name)
                row = inside[0]
            else:
                row = rows[0]
            hexid = shift(row["Hex"])
            gg = int(row["PBG"][2]) > 0
            owner = ALLEGIANCE.get(row["Allegiance"], "neutral")
        if name in FORCE_IMPERIAL:
            owner = "imperial"
        display = re.sub(r"\s*\(\d{4}\)$", "", name)
        if hexid in used_hexes:
            raise SystemExit("duplicate hex %s (%s / %s)" % (hexid, name, used_hexes[hexid]))
        used_hexes[hexid] = name
        worlds.append({
            "name": display,
            "key": name,
            "hex": hexid,
            "starport": port,
            "tech_level": tl,
            "sdb": sdb,
            "defense_battalions": defbn,
            "water": bool(water),
            "atmosphere": atm,
            "naval_base": "N" in bases,
            "scout_base": "S" in bases,
            "military_base": "M" in bases,
            "gas_giant": bool(gg),
            "zone": zone,
            "owner": owner,
            "subsector_capital": hexid in SUBSECTOR_CAPITALS,
        })

    # Kinorb appears twice on the map; disambiguate the display names.
    dupes = {}
    for w in worlds:
        dupes.setdefault(w["name"], []).append(w)
    for nm, group in dupes.items():
        if len(group) > 1:
            for w in group:
                w["name"] = "%s (%s)" % (nm, w["hex"])

    hexes = {w["hex"] for w in worlds}
    routes = []
    meta = json.load(open(os.path.join(HERE, "spinmeta.json"), encoding="utf-8"))
    for r in meta.get("Routes", []):
        if any(k in r for k in ("StartOffsetX", "StartOffsetY",
                                "EndOffsetX", "EndOffsetY")):
            continue
        a, b = shift(r["Start"]), shift(r["End"])
        if a in hexes and b in hexes and (b, a) not in routes:
            routes.append((a, b))

    data = {
        "note": "Fifth Frontier War (GDW 1981) map data; see tools/build_worlds.py",
        "hex_offset_from_canonical": list(HEX_OFFSET),
        "entry_hexes": {
            "imperial_reinforcements": "3218",   # Jae Tellona (rules say 2218: typo)
            "imperial_rimward": "2822",          # Icetina
            "sword_worlds": "1627",              # Gram
            "zhodani_reinforcements": "0708",    # Chronor
            "zhodani_additional": "0708",
        },
        "worlds": worlds,
        "xboat_routes": [list(r) for r in routes],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=False)
    print("wrote", OUT, len(worlds), "worlds,", len(routes), "xboat routes")

    # ---- consistency check against the canonical UWPs ------------------------
    def atm_class(code):
        v = int(code, 16) if code not in "X?" else 0
        if v <= 1:
            return VAC
        return BREATH if v in (5, 6, 8, 13) else TAINT

    bad = 0
    for w in worlds:
        rows = canon.get(w["key"].split(" (")[0].lower())
        if not rows or w["key"] in OFF_SECTOR:
            continue
        row = [r for r in rows if shift(r["Hex"]) == w["hex"]]
        if not row:
            continue
        uwp = row[0]["UWP"]
        if uwp[0] != w["starport"] and not (uwp[0] == "X" and w["starport"] == "X"):
            print("  starport differs: %-14s map=%s canon=%s" % (w["name"], w["starport"], uwp[0]))
            bad += 1
        if atm_class(uwp[2]) != w["atmosphere"]:
            print("  atmosphere differs: %-14s map=%-10s canon=%s" % (w["name"], w["atmosphere"], uwp[2]))
            bad += 1
        if (uwp[3] != "0") != w["water"]:
            print("  water differs: %-14s map=%s canon=%s" % (w["name"], w["water"], uwp[3]))
            bad += 1
    print("cross-check discrepancies vs canonical UWPs:", bad)


if __name__ == "__main__":
    main()
