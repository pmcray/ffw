"""The surface of Terra as a hex grid, and the geometry that goes with it.

*Invasion: Earth*'s map is the Earth "projected onto the twenty triangular
sides of an icosahedron", cut into hexes 1148 km across, with the rulebook
adding two rules that matter more than they look:

    A hex divided between two or more of these triangles is considered to be a
    single hex for all purposes.

    The eastern and western edges of the map are considered to be adjacent, and
    a half hex on one side of the map has its other half on the other side.

Both rules exist to undo the damage the paper does.  A sphere cannot be cut
into hexes and laid flat without tearing, so the printed map tears it along the
icosahedron's edges and then tells you to ignore the tears.  What the rules
describe, once those two sentences are applied, is a **hex grid on a sphere** --
and a sphere is something a computer can hold directly, with no tears to
apologise for.

So this module builds the thing the paper map is a picture of: a Goldberg
polyhedron, the dual of a subdivided icosahedron.  Every cell is a hexagon
except for exactly twelve pentagons at the icosahedron's vertices; that twelve
is not a modelling choice but a theorem -- Euler's formula forbids tiling a
sphere with hexagons alone, which is precisely why the printed map has seams.
Adjacency wraps in every direction, the poles are ordinary cells, and no hex is
half on one edge of a sheet and half on the other.

The frequency is chosen so the cells come out the size the rules say they are.
A Goldberg polyhedron of frequency ``n`` has ``10n^2 + 2`` cells; at ``n = 7``
that is 492 cells averaging about 1.04 million km^2, or 1096 km across, against
the rulebook's 1148.  ``n = 6`` (362 cells) would make them 1270 km across.
Seven is the closer fit and is what the map uses.
"""

from __future__ import annotations

import math

#: Goldberg frequency.  10n^2 + 2 cells; see the module docstring for why 7.
FREQUENCY = 7

#: Mean radius of Terra in kilometres.
RADIUS_KM = 6371.0

#: What the rulebook prints as the width of a hex.  The grid lands near it
#: rather than on it, because cell count is quantised to 10n^2 + 2.
NOMINAL_HEX_KM = 1148.0

_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _normalise(v):
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    return (x / n, y / n, z / n)


def _icosahedron():
    """The twenty triangular sides the rulebook starts from.

    Returns ``(vertices, faces)`` with vertices on the unit sphere.  Orientation
    is the standard one with two vertices at the poles, which is also how the
    printed map is laid out -- polar caps top and bottom, an equatorial band
    between.
    """
    verts = []
    for i in range(12):
        if i == 0:
            verts.append((0.0, 0.0, 1.0))
        elif i == 11:
            verts.append((0.0, 0.0, -1.0))
        else:
            # two staggered rings of five, at the latitudes where the
            # icosahedron's vertices fall
            band = (i - 1) // 5
            k = (i - 1) % 5
            lat = math.atan(0.5) * (1 if band == 0 else -1)
            lon = 2.0 * math.pi * k / 5.0 + (0.0 if band == 0 else math.pi / 5.0)
            verts.append((math.cos(lat) * math.cos(lon),
                          math.cos(lat) * math.sin(lon),
                          math.sin(lat)))
    faces = []
    top, bottom = 0, 11
    upper = list(range(1, 6))
    lower = list(range(6, 11))
    for k in range(5):
        faces.append((top, upper[k], upper[(k + 1) % 5]))
        faces.append((bottom, lower[(k + 1) % 5], lower[k]))
        faces.append((upper[k], lower[k], upper[(k + 1) % 5]))
        faces.append((upper[(k + 1) % 5], lower[k], lower[(k + 1) % 5]))
    return verts, faces


def _subdivide(frequency: int):
    """Subdivide each icosahedral face into ``frequency^2`` triangles.

    The result is a geodesic sphere; its *dual* -- one cell per vertex -- is the
    hex grid.  Vertices shared between faces are merged by rounding their
    coordinates to a key, which is exact enough here because the subdivision
    produces the same points from both sides by the same arithmetic.
    """
    verts, faces = _icosahedron()
    points: dict[tuple, int] = {}
    coords: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []

    def index_of(v):
        v = _normalise(v)
        key = (round(v[0], 9), round(v[1], 9), round(v[2], 9))
        if key not in points:
            points[key] = len(coords)
            coords.append(v)
        return points[key]

    for a, b, c in faces:
        va, vb, vc = verts[a], verts[b], verts[c]
        # barycentric lattice over the face, then projected onto the sphere
        grid: dict[tuple[int, int], int] = {}
        for i in range(frequency + 1):
            for j in range(frequency + 1 - i):
                k = frequency - i - j
                p = tuple((i * va[d] + j * vb[d] + k * vc[d]) / frequency
                          for d in range(3))
                grid[(i, j)] = index_of(p)
        for i in range(frequency):
            for j in range(frequency - i):
                triangles.append((grid[(i, j)], grid[(i + 1, j)], grid[(i, j + 1)]))
                if i + j < frequency - 1:
                    triangles.append((grid[(i + 1, j)], grid[(i + 1, j + 1)],
                                      grid[(i, j + 1)]))
    return coords, triangles


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _outlines(coords, triangles):
    """The corners of each cell, in order around it.

    The dual construction gives one cell per geodesic *vertex*; the corners of
    that cell are the centres of the triangles meeting at the vertex, which is
    what makes the twelve pentagons pentagons.  They come out of the
    subdivision in no particular order, so they are sorted by angle in the
    tangent plane at the cell's own centre -- a polygon whose corners are not
    in order around it draws as a bow tie.
    """
    incident: list[list] = [[] for _ in coords]
    for tri in triangles:
        middle = _normalise(tuple(sum(coords[v][d] for v in tri) / 3.0
                                  for d in range(3)))
        for vertex in tri:
            incident[vertex].append(middle)
    outlines = []
    for cell, corners in enumerate(incident):
        centre = coords[cell]
        # any vector not parallel to the centre gives a tangent basis
        ref = (0.0, 0.0, 1.0) if abs(centre[2]) < 0.9 else (1.0, 0.0, 0.0)
        east = _normalise(_cross(ref, centre))
        north = _cross(centre, east)
        outlines.append(tuple(sorted(
            corners, key=lambda p: math.atan2(_dot(p, north), _dot(p, east)))))
    return outlines


def _build():
    """The dual: one cell per geodesic vertex, adjacency from shared edges."""
    coords, triangles = _subdivide(FREQUENCY)
    neighbours: list[set] = [set() for _ in coords]
    for a, b, c in triangles:
        neighbours[a].update((b, c))
        neighbours[b].update((a, c))
        neighbours[c].update((a, b))
    return coords, [sorted(n) for n in neighbours], _outlines(coords, triangles)


_COORDS, _NEIGHBOURS, _OUTLINES = _build()

#: Every cell on the map, in a stable order.  Cell ids are opaque: use
#: ``lat_lon`` to place them and ``nearest`` to find them.
CELLS = tuple(range(len(_COORDS)))

#: The twelve pentagons, at the icosahedron's own vertices.  Euler's formula
#: says a sphere cannot be tiled with hexagons alone; these twelve are where
#: the printed map has to tear.
PENTAGONS = frozenset(c for c in CELLS if len(_NEIGHBOURS[c]) == 5)


def cell_count() -> int:
    return len(CELLS)


def neighbours(cell: int) -> tuple[int, ...]:
    """The five or six cells sharing an edge with ``cell``."""
    return tuple(_NEIGHBOURS[cell])


def unit_vector(cell: int) -> tuple[float, float, float]:
    return _COORDS[cell]


def outline(cell: int) -> tuple:
    """The cell's corners as unit vectors, in order around it.

    Five of them for the twelve pentagons and six for everything else, which is
    the one visible consequence of Euler's formula on the finished map.
    """
    return _OUTLINES[cell]


def lat_lon(cell: int) -> tuple[float, float]:
    """Latitude and longitude in degrees; longitude in (-180, 180]."""
    x, y, z = _COORDS[cell]
    lat = math.degrees(math.asin(max(-1.0, min(1.0, z))))
    lon = math.degrees(math.atan2(y, x))
    if lon <= -180.0:
        lon += 360.0
    return lat, lon


def nearest(lat: float, lon: float) -> int:
    """The cell containing a point, by great-circle distance to cell centres."""
    la, lo = math.radians(lat), math.radians(lon)
    target = (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo),
              math.sin(la))
    best, best_dot = 0, -2.0
    for c in CELLS:
        x, y, z = _COORDS[c]
        dot = x * target[0] + y * target[1] + z * target[2]
        if dot > best_dot:
            best, best_dot = c, dot
    return best


def distance(a: int, b: int) -> int:
    """Distance in hexes: the fewest steps from ``a`` to ``b``.

    Grid distance rather than great-circle, because every rule that measures
    range -- a PD unit's three hexes, a supply line's five, a grav unit's ten
    movement points -- counts hexes.
    """
    if a == b:
        return 0
    seen = {a}
    frontier = [a]
    steps = 0
    while frontier:
        steps += 1
        nxt = []
        for cell in frontier:
            for n in _NEIGHBOURS[cell]:
                if n == b:
                    return steps
                if n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier = nxt
    raise ValueError("disconnected grid")   # pragma: no cover


def within(origin: int, radius: int) -> tuple[int, ...]:
    """Every cell within ``radius`` hexes of ``origin``, excluding it."""
    seen = {origin}
    frontier = [origin]
    out: list[int] = []
    for _ in range(radius):
        nxt = []
        for cell in frontier:
            for n in _NEIGHBOURS[cell]:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
                    nxt.append(n)
        frontier = nxt
    return tuple(out)


def cell_area_km2() -> float:
    """Mean area of a cell, from the sphere's area and the cell count."""
    return 4.0 * math.pi * RADIUS_KM ** 2 / len(CELLS)


def cell_width_km() -> float:
    """Width across flats of a regular hexagon of the mean cell area."""
    return math.sqrt(cell_area_km2() / (math.sqrt(3.0) / 2.0))


def great_circle_km(a: int, b: int) -> float:
    ax, ay, az = _COORDS[a]
    bx, by, bz = _COORDS[b]
    dot = max(-1.0, min(1.0, ax * bx + ay * by + az * bz))
    return RADIUS_KM * math.acos(dot)
