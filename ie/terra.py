"""Terra's surface: which hex is land, and what kind of land it is.

The printed map is a painting of the Earth, and the only way to get it into a
program is to describe the Earth somewhere.  This module does it as data --
land as longitude spans per latitude band, then named regions layered on top --
rather than by reading pixels off a scan of the map, for three reasons.  The
scan is a photograph of a folded paper sheet, so its colours are unreliable at
the edges; the icosahedral net would have to be un-projected before a pixel
could be turned into a latitude; and a table of coastlines can be read and
argued with, which a colour threshold cannot.

The resolution that matters is the hex.  A cell is about 1100 km across -- ten
degrees of latitude -- so the coastlines below are deliberately coarse.  At
this scale Britain is a single hex shared with the North Sea, and the Isthmus
of Panama does not exist.  That is a property of the game, not of the model:
the printed map has the same problem and solves it the same way.

Everything here is checked against the rulebook's own counts, which are
specific enough to be a real test: sixty-one urban hexes, three starports, and
armies deployed to Africa, Asia, North America and South America.
"""

from __future__ import annotations

import json
import os

from . import hexmap

#: Terrain types, exactly as the map's terrain key lists them.
CLEAR = "clear"
URBAN = "urban"
STARPORT = "starport"
TUNDRA = "tundra"                 # tundra / ice sheet wilderness
FOREST = "forest"                 # forest / jungle wilderness
MOUNTAIN = "mountain"             # mountainous wilderness
DESERT = "desert"
SEA = "sea"
ISLANDS = "islands"
PERMANENT_ICE = "permanent_ice"   # permanent sea ice sheet
SEASONAL_ICE = "seasonal_ice"     # seasonal sea ice sheet

#: Land, for every rule that says "may not end its movement in an all-sea hex".
LAND = frozenset({CLEAR, URBAN, STARPORT, TUNDRA, FOREST, MOUNTAIN, DESERT,
                  ISLANDS})

#: Where a guerrilla unit "is allowed a special advantage for the purposes of
#: combat": urban, desert, and any type of wilderness.
GUERRILLA_COVER = frozenset({URBAN, STARPORT, DESERT, TUNDRA, FOREST, MOUNTAIN})

#: Hexes that count as being in winter whatever the season -- rule 5's climate
#: paragraph: "Tundra/ice sheet hexes and sea ice sheet hexes are treated as
#: being in winter regardless of actual season."
ALWAYS_WINTER = frozenset({TUNDRA, PERMANENT_ICE, SEASONAL_ICE})

#: A SDB wing hides in "any sea hex that contains no other type of terrain
#: (including sea ice sheet terrain)" -- so plain sea only, not islands, not
#: ice.
DEEP_SEA = frozenset({SEA})


# --------------------------------------------------------------------------
# the Earth, coarsely
# --------------------------------------------------------------------------
#: Land as (latitude band) -> list of (west, east) longitude spans.  Spans that
#: cross the antimeridian are written with west > east and handled on lookup.
#: Bands are ten degrees, matching the hex.
_LAND_BANDS: dict[tuple[int, int], list[tuple[float, float]]] = {
    # The Arctic Ocean is water; only Greenland's tip reaches this far.
    (80, 90): [],
    (70, 80): [(-105, -80), (-55, -25), (55, 115)],             # Canadian arctic,
                                                                # Greenland, Taimyr
    (60, 70): [(-165, -95), (-55, -25), (-25, -14), (5, 180)],  # Alaska and Canada
                                                                # west of Hudson Bay,
                                                                # Greenland, Iceland,
                                                                # Fennoscandia, Siberia
    (50, 60): [(-125, -95), (-80, -56), (-11, 180)],            # Canada, Labrador,
                                                                # Britain, Europe,
                                                                # Russia to Kamchatka
    (40, 50): [(-125, -66), (-10, 60), (60, 145)],              # USA, Europe,
                                                                # central Asia, Japan
    (30, 40): [(-122, -76), (-9, 48), (48, 122), (128, 141)],   # USA, Mediterranean,
                                                                # Middle East, China,
                                                                # Korea and Japan
    (20, 30): [(-112, -99), (-84, -80), (-16, 57), (68, 90),
               (98, 121)],                                      # Mexico, Florida,
                                                                # Sahara, Arabia,
                                                                # India, south China
    (10, 20): [(-104, -86), (-16, 48), (73, 88), (96, 108)],    # Central America,
                                                                # Sahel, India,
                                                                # Indochina
    (0, 10): [(-78, -52), (-12, 46), (100, 118)],               # northern South
                                                                # America, equatorial
                                                                # Africa, Borneo
    (-10, 0): [(-76, -37), (10, 40), (100, 118), (131, 148)],   # Amazon, Congo,
                                                                # Indonesia, New Guinea
    (-20, -10): [(-72, -37), (12, 40), (44, 49), (115, 148)],   # Brazil, Africa,
                                                                # Madagascar, Australia
    (-30, -20): [(-70, -42), (14, 35), (114, 152)],             # Brazil, Africa,
                                                                # Australia
    (-40, -30): [(-71, -55), (17, 32), (116, 150), (170, 177)], # Argentina, Cape,
                                                                # Australia, New Zealand
    (-50, -40): [(-73, -64), (166, 174)],                       # Patagonia, New Zealand
    (-60, -50): [(-74, -67)],                                   # Tierra del Fuego
    # Sixty to seventy south is the Southern Ocean; only the Antarctic
    # Peninsula and a few coastal salients reach north of the seventieth.
    (-70, -60): [(-64, -56), (95, 115)],
    (-80, -70): [(-180, 180)],                                  # Antarctica
    (-90, -80): [(-180, 180)],                                  # Antarctica
}

#: Islands: real land too small to fill a hex, but a hex all the same.  The map
#: gives these their own terrain type, and the rules let troops end movement
#: there, so they are the stepping stones across the Pacific.
_ISLANDS = [
    (21, -158, "Hawaii"), (-18, 178, "Fiji"), (-17, -149, "Tahiti"),
    (13, 145, "Marianas"), (7, 134, "Palau"), (-6, 155, "Solomons"),
    (-22, 166, "New Caledonia"), (-37, 178, "Chatham approaches"),
    (28, -177, "Midway"), (-15, -5, "Ascension"), (-37, -12, "Tristan"),
    (-54, -37, "South Georgia"), (-49, 69, "Kerguelen"), (-11, 105, "Christmas"),
    (4, 73, "Maldives"), (-20, 57, "Mascarenes"), (65, -22, "Iceland"),
    (39, -28, "Azores"), (16, -24, "Cape Verde"), (18, -66, "Antilles"),
    (25, -77, "Bahamas"), (-8, -14, "St Helena rise"),
]

#: Regions, applied in order after the base climate classification.  A region
#: is a name, a terrain type, and a list of (lat0, lat1, lon0, lon1) boxes.
#: Later entries win, so the ordering is deliberate: physical geography first,
#: then the cities on top of it.
_REGIONS: list[tuple[str, str, list[tuple[float, float, float, float]]]] = [
    ("Sahara", DESERT, [(15, 30, -14, 34)]),
    ("Arabia", DESERT, [(15, 30, 34, 60)]),
    ("Iran and Thar", DESERT, [(24, 36, 52, 72)]),
    ("Gobi and Taklamakan", DESERT, [(36, 46, 75, 112)]),
    ("Kalahari and Namib", DESERT, [(-30, -18, 12, 25)]),
    ("Australian interior", DESERT, [(-30, -19, 118, 142)]),
    ("Atacama", DESERT, [(-28, -18, -71, -66)]),
    ("Great Basin", DESERT, [(32, 42, -119, -110)]),
    ("Amazonia", FOREST, [(-10, 4, -74, -50)]),
    ("Congo", FOREST, [(-6, 6, 10, 30)]),
    ("Guinea coast", FOREST, [(4, 12, -14, 10)]),
    ("Indochina and the Indies", FOREST, [(-9, 22, 95, 141)]),
    ("Taiga", FOREST, [(52, 66, 30, 140)]),
    ("Boreal Canada", FOREST, [(50, 60, -130, -60)]),
    ("New Guinea", FOREST, [(-10, -1, 131, 150)]),
    ("Rockies", MOUNTAIN, [(36, 52, -125, -114)]),
    ("Andes", MOUNTAIN, [(-40, 8, -80, -66)]),
    ("Himalaya and Tibet", MOUNTAIN, [(26, 40, 72, 100)]),
    ("Caucasus and Zagros", MOUNTAIN, [(32, 44, 40, 56)]),
    ("Alps", MOUNTAIN, [(43, 48, 6, 16)]),
    ("East African rift", MOUNTAIN, [(-4, 12, 30, 44)]),
    ("Greenland ice", TUNDRA, [(60, 90, -60, -18)]),
    ("Canadian arctic", TUNDRA, [(66, 84, -136, -60)]),
    ("Siberian arctic", TUNDRA, [(66, 82, 55, 180)]),
    ("Fennoscandian arctic", TUNDRA, [(66, 72, 4, 55)]),
    ("Alaskan arctic", TUNDRA, [(66, 72, -168, -136)]),
    ("Antarctica", TUNDRA, [(-90, -60, -180, 180)]),
    # ---- the cities -----------------------------------------------------
    # Terra in 1002 is an old, crowded, heavily industrialised world; the
    # printed map is red across every region that is densely settled today.
    ("Eastern North America", URBAN, [(30, 48, -92, -66)]),
    ("Western North America", URBAN, [(30, 52, -125, -112)]),
    ("Great Lakes", URBAN, [(38, 48, -92, -76)]),
    ("Mexico and Central America", URBAN, [(14, 26, -106, -88)]),
    ("Caribbean rim", URBAN, [(8, 22, -84, -60)]),
    ("Brazilian coast", URBAN, [(-26, -4, -50, -34)]),
    ("River Plate", URBAN, [(-38, -26, -62, -52)]),
    ("Andean cities", URBAN, [(-24, 12, -80, -66)]),
    ("Western Europe", URBAN, [(43, 56, -10, 16)]),
    ("Britain", URBAN, [(50, 60, -11, 2)]),
    ("Central and eastern Europe", URBAN, [(43, 56, 16, 40)]),
    ("Mediterranean", URBAN, [(34, 45, -8, 30)]),
    ("Nile", URBAN, [(22, 33, 28, 36)]),
    ("Levant and Mesopotamia", URBAN, [(26, 40, 32, 52)]),
    ("Gulf", URBAN, [(22, 30, 46, 58)]),
    ("West Africa", URBAN, [(4, 16, -16, 16)]),
    ("East Africa", URBAN, [(-8, 13, 28, 45)]),
    ("Southern Africa", URBAN, [(-34, -20, 16, 34)]),
    ("Maghreb", URBAN, [(30, 38, -8, 12)]),
    ("India", URBAN, [(8, 30, 70, 90)]),
    ("Indochina", URBAN, [(6, 24, 94, 112)]),
    ("Indonesia", URBAN, [(-9, 2, 104, 120)]),
    ("Eastern China", URBAN, [(20, 44, 104, 124)]),
    ("Korea and Japan", URBAN, [(30, 44, 126, 142)]),
    ("Central Asia", URBAN, [(38, 54, 56, 86)]),
    ("Russia west of the Urals", URBAN, [(48, 60, 28, 56)]),
    ("Siberian industry", URBAN, [(52, 60, 82, 120)]),
    ("Australian coast", URBAN, [(-40, -22, 138, 154)]),
    ("Western Australia", URBAN, [(-36, -28, 113, 120)]),
    ("New Zealand", URBAN, [(-46, -34, 165, 179)]),
]

#: How many urban hexes the Solomani start with.  This is not a modelling
#: choice: rule 6 states it outright -- "The Solomani player has 61 urban hexes
#: at the start of the game" -- and the number is load-bearing, because the
#: Solomani draw one replacement point per ungarrisoned urban hex each turn and
#: the game ends when fewer than ten remain to them.
URBAN_HEXES = 61


#: The three starports, with the names printed on the map.  The order of battle
#: puts one PD corps on each, and the counters name them LG, AECO and SP.
STARPORTS = [
    ("Starport Phoenix", 33.4, -112.1),
    ("AECO Starport", 0.4, 32.5),
    ("LaGrange Starport", -31.9, 115.9),
]

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CACHE = os.path.join(_DATA, "terrain.json")


# --------------------------------------------------------------------------
def _in_span(lon: float, west: float, east: float) -> bool:
    if west <= east:
        return west <= lon <= east
    return lon >= west or lon <= east          # crosses the antimeridian


def is_land(lat: float, lon: float) -> bool:
    for (lo, hi), spans in _LAND_BANDS.items():
        if lo <= lat < hi:
            return any(_in_span(lon, w, e) for w, e in spans)
    return False


def _in_box(lat, lon, box) -> bool:
    lat0, lat1, lon0, lon1 = box
    if not (lat0 <= lat <= lat1):
        return False
    return _in_span(lon, lon0, lon1)


def _sea_terrain(lat: float) -> str:
    """Open water, iced according to latitude.

    The map distinguishes permanent from seasonal sea ice, and the climate rule
    turns the seasonal band into plain sea outside its hemisphere's winter.
    """
    if abs(lat) >= 75:
        return PERMANENT_ICE
    if abs(lat) >= 62:
        return SEASONAL_ICE
    return SEA


def classify(lat: float, lon: float) -> str:
    """The terrain of a point, before starports are placed."""
    if not is_land(lat, lon):
        for ilat, ilon, _name in _ISLANDS:
            if abs(ilat - lat) < 4 and abs(((ilon - lon + 180) % 360) - 180) < 5:
                return ISLANDS
        return _sea_terrain(lat)
    terrain = CLEAR
    for _name, kind, boxes in _REGIONS:
        if any(_in_box(lat, lon, b) for b in boxes):
            terrain = kind
    return terrain


def _bare_terrain(lat: float, lon: float) -> str:
    """The terrain of a land point ignoring the cities on top of it."""
    terrain = CLEAR
    for _name, kind, boxes in _REGIONS:
        if kind == URBAN:
            continue
        if any(_in_box(lat, lon, b) for b in boxes):
            terrain = kind
    return terrain


def settlement_score(lat: float, lon: float) -> float:
    """How built-up a point is: how many urban regions claim it, plus a tiebreak.

    The urban regions above overlap deliberately -- the Great Lakes sit inside
    Eastern North America, the Mediterranean inside Western Europe -- so a cell
    claimed by three of them is more densely settled than one claimed by a
    single region.  That ordering is what lets the map be trimmed to the
    rulebook's count without an arbitrary choice about *which* hexes go.

    The tiebreak is latitude: where two hexes are equally claimed, the more
    temperate one is the more settled.  It exists to make the result
    deterministic, not because it says anything about Terra.
    """
    claims = sum(1 for _name, kind, boxes in _REGIONS if kind == URBAN
                 and any(_in_box(lat, lon, b) for b in boxes))
    return claims - abs(lat) / 1000.0


def build() -> dict:
    """Classify every cell, trim the cities to 61, then place the starports.

    Three steps, in that order, and the order matters.  Geography first, so the
    deserts and jungles are where they are.  Then the cities: the region boxes
    describe *where* Terra is built up, but the rulebook fixes *how much* at 61
    hexes, so the most-claimed 61 land hexes keep their cities and the rest
    revert to the terrain underneath them.  A grid of 492 cells will not land
    on 61 by accident, and the count matters too much to leave to luck.

    Starports go on last and replace whatever was there, because the map draws
    a starport as its own terrain type rather than as a marker on top of one.
    """
    terrain, scores = {}, {}
    for cell in hexmap.CELLS:
        lat, lon = hexmap.lat_lon(cell)
        terrain[cell] = classify(lat, lon)
        scores[cell] = settlement_score(lat, lon)

    # the starport hexes are settled too, but the rulebook counts them
    # separately -- 61 urban hexes *and* three starports -- so they are taken
    # out of the running before the cities are counted
    ports = {}
    for name, lat, lon in STARPORTS:
        ports[hexmap.nearest(lat, lon)] = name

    candidates = sorted((c for c in hexmap.CELLS
                         if terrain[c] in LAND and c not in ports),
                        key=lambda c: (-scores[c], c))
    keep = set(candidates[:URBAN_HEXES])
    for cell in hexmap.CELLS:
        lat, lon = hexmap.lat_lon(cell)
        if cell in keep:
            terrain[cell] = URBAN
        elif terrain[cell] == URBAN:
            # a city the trim did not keep: fall back to the land under it
            terrain[cell] = _bare_terrain(lat, lon)
    for cell in ports:
        terrain[cell] = STARPORT
    return {"terrain": terrain, "starports": ports}


def load() -> dict:
    """The map, from the cache if it is there and freshly built if not."""
    if os.path.exists(_CACHE):
        with open(_CACHE, encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("frequency") == hexmap.FREQUENCY:
            return {"terrain": {int(k): v for k, v in raw["terrain"].items()},
                    "starports": {int(k): v for k, v in raw["starports"].items()}}
    return build()


def save(path: str | None = None) -> str:
    built = build()
    payload = {"frequency": hexmap.FREQUENCY,
               "cells": hexmap.cell_count(),
               "terrain": {str(k): v for k, v in built["terrain"].items()},
               "starports": {str(k): v for k, v in built["starports"].items()}}
    path = path or _CACHE
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
    return path


#: Continent names, for the Solomani deployment rule -- "one army is placed in
#: any urban hex on each of the following continents: Africa, Asia, North
#: America, and South America".
_CONTINENTS = [
    ("North America", [(15, 84, -168, -52)]),
    ("South America", [(-56, 15, -82, -34)]),
    ("Africa", [(-35, 37, -18, 52)]),
    ("Europe", [(36, 72, -11, 40)]),
    ("Asia", [(-11, 78, 40, 180)]),
    ("Australia", [(-48, -10, 112, 180)]),
    ("Antarctica", [(-90, -60, -180, 180)]),
]


def continent(lat: float, lon: float) -> str | None:
    """Which continent a point is on, or ``None`` for open ocean.

    Europe is listed separately from Asia and searched first, because the
    deployment rule names Asia and would otherwise put a Solomani field army in
    Paris and call it Asia.
    """
    for name, boxes in _CONTINENTS:
        if any(_in_box(lat, lon, b) for b in boxes):
            return name
    return None
