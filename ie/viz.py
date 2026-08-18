"""Drawing Terra.

*Fifth Frontier War* draws as a flat sheet of hexes because that is what its
map is.  *Invasion: Earth* cannot: its map is a Goldberg polyhedron, 492 cells
on a sphere, and the printed game only gets it onto paper by tearing it along
the icosahedron's edges and then telling the players to ignore the tears.  A
picture that reproduced the tears would be reproducing the compromise rather
than the game.

So ``draw_map`` draws the planet as a planet: two orthographic hemispheres, one
centred on the fighting and one on its antipode, so that all 492 cells are on
the page at once and adjacency across the limb is where the eye expects it.
Next to them is the part of this game that is not on the planet at all -- the
four space boxes, where the whole Imperial army starts and most of it usually
still is.

``GameRecorder`` keeps a light snapshot of every turn, the same shape as
``ffw.viz``'s, so a finished campaign can be replayed or plotted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from . import hexmap, terra
from .state import (CLOSE_ORBIT, DEEP_SPACE, FAR_ORBIT, GameState, IMPERIAL,
                    LUNA, OUT_SYSTEM, SOLOMANI)

#: The same dark chart palette ``ffw.viz`` uses, so the two games look like one
#: project, with a terrain ramp added for the surface.
COLORS = {
    "background": "#11151c",
    "imperial": "#d94f4f",
    "solomani": "#4f8fd9",
    "text": "#e8ecf2",
    "faint": "#7f8894",
    "grid": "#2c3440",
    "garrison": "#f2c14e",
}

TERRAIN_COLOR = {
    terra.SEA: "#16293d",
    terra.ISLANDS: "#24506b",
    terra.PERMANENT_ICE: "#d7e2ec",
    terra.SEASONAL_ICE: "#a8bacc",
    terra.CLEAR: "#4a6b41",
    terra.FOREST: "#2e4a2b",
    terra.MOUNTAIN: "#5c5449",
    terra.DESERT: "#8a7548",
    terra.TUNDRA: "#59636e",
    terra.URBAN: "#a89b7d",
    terra.STARPORT: "#c9b06a",
}

SIDE_COLOR = {IMPERIAL: COLORS["imperial"], SOLOMANI: COLORS["solomani"]}

#: The space boxes in the order the rules run them, out to in.
BOXES = (OUT_SYSTEM, DEEP_SPACE, FAR_ORBIT, CLOSE_ORBIT)
BOX_LABEL = {OUT_SYSTEM: "out-system", DEEP_SPACE: "deep space",
             FAR_ORBIT: "far orbit", CLOSE_ORBIT: "close orbit"}


# --------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Everything the renderer needs for one turn."""
    turn: int
    garrisoned: set
    surface: dict                 # cell -> {side: combat factors}
    bases: list
    orbit: dict                   # box -> {"squadrons": n, "cargo": factors}
    afloat: float
    taken: int
    points: int
    victory: str
    log: list = field(default_factory=list)


def snapshot(state: GameState) -> Snapshot:
    surface: dict = {}
    for unit in state.surface.values():
        if unit.carrier is not None or unit.dead:
            continue
        if not isinstance(unit.location, int):
            continue
        here = surface.setdefault(unit.location, {})
        here[unit.side] = here.get(unit.side, 0.0) + unit.current
    orbit = {}
    for box in BOXES:
        squadrons = [u for u in state.naval.values()
                     if u.location == box and u.side == IMPERIAL]
        cargo = sum(sum(_cargo_factor(state, uid) for uid in u.cargo)
                    for u in squadrons)
        orbit[box] = {"squadrons": len(squadrons), "cargo": cargo,
                      "solomani": len([u for u in state.naval.values()
                                       if u.location == box
                                       and u.side == SOLOMANI])}
    afloat = sum(u.current for u in state.surface.values()
                 if u.side == IMPERIAL and u.carrier is not None and not u.dead)
    return Snapshot(
        turn=state.turn,
        garrisoned=set(state.garrisoned),
        surface=surface,
        bases=[b.location for b in state.bases.values()
               if b.carrier is None and not b.dead
               and isinstance(b.location, int)],
        orbit=orbit,
        afloat=afloat,
        taken=len(state.geometry.urban) - len(state.solomani_urban()),
        points=state.victory_points(),
        victory=state.result or state.victory_level(),
        log=list(state.log[-6:]))


def _cargo_factor(state, uid) -> float:
    unit = state.surface.get(uid)
    if unit is not None:
        return unit.current
    return 100.0 if uid in state.bases else 0.0


class GameRecorder:
    """Collects one snapshot per turn; pass ``recorder`` as ``on_turn``."""

    def __init__(self):
        self.frames: list[Snapshot] = []

    def __call__(self, state: GameState) -> None:
        self.frames.append(snapshot(state))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index) -> Snapshot:
        return self.frames[index]


# --------------------------------------------------------------------------
#  the projection
# --------------------------------------------------------------------------
def _basis(centre):
    """East and north unit vectors in the tangent plane at ``centre``."""
    ref = (0.0, 0.0, 1.0) if abs(centre[2]) < 0.9 else (1.0, 0.0, 0.0)
    east = hexmap._normalise(hexmap._cross(ref, centre))
    return east, hexmap._cross(centre, east)


def _project(point, centre, east, north):
    """Orthographic projection: the hemisphere facing the viewer, flat."""
    depth = hexmap._dot(point, centre)
    return hexmap._dot(point, east), hexmap._dot(point, north), depth


def _clipped_outline(cell, centre, east, north):
    """The cell's corners, with anything over the limb pulled back onto it.

    A cell straddling the edge of the visible hemisphere has corners behind the
    planet; projected as they are, they fold the polygon inside out.  Each such
    corner is moved along the great circle towards the cell's own centre until
    it reaches the limb, which is where it would be seen from.
    """
    middle = hexmap.unit_vector(cell)
    corners = []
    for corner in hexmap.outline(cell):
        if hexmap._dot(corner, centre) < 0.0:
            lo, hi = 0.0, 1.0
            for _ in range(12):                # bisect onto the limb
                mid = (lo + hi) / 2.0
                point = hexmap._normalise(tuple(
                    corner[d] + (middle[d] - corner[d]) * mid for d in range(3)))
                if hexmap._dot(point, centre) < 0.0:
                    lo = mid
                else:
                    hi = mid
            corner = hexmap._normalise(tuple(
                corner[d] + (middle[d] - corner[d]) * hi for d in range(3)))
        x, y, _depth = _project(corner, centre, east, north)
        corners.append((x, y))
    return corners


def focus(state: GameState, frame: Snapshot | None = None):
    """Where to point the camera: the fighting, and the far side of it.

    The invasion happens in one place and the rest of the planet is scenery, so
    the first hemisphere is centred on the Imperial army if there is one ashore
    and on the heaviest Solomani concentration if there is not.
    """
    surface = frame.surface if frame is not None else snapshot(state).surface
    weights: dict = {}
    for cell, sides in surface.items():
        if IMPERIAL in sides:
            weights[cell] = weights.get(cell, 0.0) + sides[IMPERIAL] * 4.0
        weights[cell] = weights.get(cell, 0.0) + sides.get(SOLOMANI, 0.0)
    if not weights:
        anchor = next(iter(sorted(state.geometry.urban)))
    else:
        anchor = max(weights, key=lambda c: (weights[c], -c))
    near = hexmap.unit_vector(anchor)
    return near, tuple(-v for v in near)


# --------------------------------------------------------------------------
#  the map
# --------------------------------------------------------------------------
def draw_globe(state: GameState, centre, frame: Snapshot | None = None,
               ax=None, label: str | None = None):
    """One hemisphere: terrain, cities, garrisons, and what is standing on it."""
    if frame is None:
        frame = snapshot(state)
    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 6))
    geo = state.geometry
    east, north = _basis(centre)

    ax.add_patch(Circle((0, 0), 1.0, facecolor="#0b1017",
                        edgecolor=COLORS["grid"], linewidth=1.0, zorder=0))
    for cell in geo.cells:
        middle = hexmap.unit_vector(cell)
        if hexmap._dot(middle, centre) <= 0.0:
            continue
        terrain = geo.terrain[cell]
        ax.add_patch(Polygon(
            _clipped_outline(cell, centre, east, north), closed=True,
            facecolor=TERRAIN_COLOR.get(terrain, "#333a44"),
            edgecolor="#0d1218", linewidth=0.25, zorder=1))

    def place(cell):
        x, y, _d = _project(hexmap.unit_vector(cell), centre, east, north)
        return x, y

    for cell in sorted(geo.urban | geo.starports):
        if hexmap._dot(hexmap.unit_vector(cell), centre) <= 0.0:
            continue
        x, y = place(cell)
        held = cell in frame.garrisoned
        ax.plot(x, y, marker="s" if cell in geo.starports else "o",
                markersize=4.0 if cell in geo.starports else 2.6,
                markerfacecolor=COLORS["garrison"] if held else "#e6dcc0",
                markeredgecolor=COLORS["imperial"] if held else "none",
                markeredgewidth=0.9, zorder=3)

    for cell in frame.bases:
        if hexmap._dot(hexmap.unit_vector(cell), centre) <= 0.0:
            continue
        x, y = place(cell)
        ax.plot(x, y, marker="P", markersize=6.0, color=COLORS["imperial"],
                markeredgecolor="#11151c", markeredgewidth=0.6, zorder=4)

    for cell, sides in sorted(frame.surface.items()):
        if hexmap._dot(hexmap.unit_vector(cell), centre) <= 0.0:
            continue
        x, y = place(cell)
        for i, side in enumerate((IMPERIAL, SOLOMANI)):
            strength = sides.get(side, 0.0)
            if strength <= 0:
                continue
            size = 3.0 + 9.0 * math.sqrt(strength / 1000.0)
            ax.plot(x + (-0.012 if i == 0 else 0.012), y,
                    marker="^" if side == IMPERIAL else "v",
                    markersize=size, color=SIDE_COLOR[side], alpha=0.9,
                    markeredgecolor="#11151c", markeredgewidth=0.5, zorder=5)

    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    if label:
        ax.set_title(label, color=COLORS["faint"], fontsize=9)
    return ax


def draw_map(state: GameState, frame: Snapshot | None = None,
             figsize=(13, 6.5)):
    """The whole planet, both hemispheres, with the space boxes beside it."""
    if frame is None:
        frame = snapshot(state)
    near, far = focus(state, frame)
    fig = plt.figure(figsize=figsize, facecolor=COLORS["background"])
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.62), wspace=0.02)
    for i, (centre, label) in enumerate(((near, "the theatre"),
                                         (far, "the far side"))):
        ax = fig.add_subplot(grid[0, i])
        ax.set_facecolor(COLORS["background"])
        draw_globe(state, centre, frame, ax=ax, label=label)

    panel = fig.add_subplot(grid[0, 2])
    panel.set_facecolor(COLORS["background"])
    panel.axis("off")
    _draw_panel(panel, state, frame)

    fig.suptitle("Invasion: Earth — turn %d, %d of %d cities taken"
                 % (frame.turn, frame.taken, len(state.geometry.urban)),
                 color=COLORS["text"], fontsize=13)
    return fig


def _draw_panel(ax, state, frame):
    """The space boxes, the scoreboard, and a legend, as text."""
    lines = ["THE SPACE BOXES", ""]
    for box in BOXES:
        entry = frame.orbit[box]
        text = "  %-12s %2d sq" % (BOX_LABEL[box], entry["squadrons"])
        if entry["cargo"]:
            text += "  %4.0f aboard" % entry["cargo"]
        if entry["solomani"]:
            text += "  (%d Solomani)" % entry["solomani"]
        lines.append(text)
    hidden = sum(1 for u in state.naval.values()
                 if u.is_sdb and u.hidden and u.side == SOLOMANI)
    lines += ["", "  %d SDB wings still hidden" % hidden, "", "THE SCOREBOARD", ""]
    ashore = sum(sides.get(IMPERIAL, 0.0) for sides in frame.surface.values())
    solomani = sum(sides.get(SOLOMANI, 0.0) for sides in frame.surface.values())
    out_system = sum(u.current for u in state.surface.values()
                     if u.side == IMPERIAL and not u.dead
                     and u.carrier is None and u.location == OUT_SYSTEM)
    lines += [
        "  Imperial ashore     %5.0f" % ashore,
        "  Imperial afloat     %5.0f" % frame.afloat,
        "  Imperial out-system %5.0f" % out_system,
        "  Solomani on Terra   %5.0f" % solomani,
        "  cities garrisoned   %5d" % frame.taken,
        "  victory points      %5d" % frame.points,
        "  %s" % frame.victory,
    ]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", family="monospace", fontsize=8.5,
            color=COLORS["text"])
    handles = [
        Line2D([], [], marker="^", linestyle="none", color=COLORS["imperial"],
               label="Imperial troops"),
        Line2D([], [], marker="v", linestyle="none", color=COLORS["solomani"],
               label="Solomani troops"),
        Line2D([], [], marker="P", linestyle="none", color=COLORS["imperial"],
               label="Imperial base"),
        Line2D([], [], marker="o", linestyle="none", color="#e6dcc0",
               label="city"),
        Line2D([], [], marker="o", linestyle="none", color=COLORS["garrison"],
               label="city garrisoned"),
        Line2D([], [], marker="s", linestyle="none", color=TERRAIN_COLOR[
            terra.STARPORT], label="starport"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8,
              labelcolor=COLORS["faint"], bbox_to_anchor=(0.0, 0.0))


# --------------------------------------------------------------------------
def plot_history(recorder: GameRecorder, ax=None, figsize=(10, 4)):
    """The campaign as three curves: what landed, what died, what was held."""
    if ax is None:
        _fig, ax = plt.subplots(figsize=figsize, facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    turns = [f.turn for f in recorder.frames]
    ashore = [sum(s.get(IMPERIAL, 0.0) for s in f.surface.values())
              for f in recorder.frames]
    solomani = [sum(s.get(SOLOMANI, 0.0) for s in f.surface.values())
                for f in recorder.frames]
    ax.plot(turns, ashore, color=COLORS["imperial"], label="Imperial ashore")
    ax.plot(turns, solomani, color=COLORS["solomani"], label="Solomani on Terra")
    ax.set_xlabel("turn", color=COLORS["faint"])
    ax.set_ylabel("combat factors", color=COLORS["faint"])
    ax.tick_params(colors=COLORS["faint"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])

    cities = ax.twinx()
    cities.plot(turns, [f.taken for f in recorder.frames],
                color=COLORS["garrison"], linestyle="--", label="cities taken")
    cities.set_ylabel("cities garrisoned", color=COLORS["garrison"])
    cities.tick_params(colors=COLORS["garrison"])
    handles, labels = ax.get_legend_handles_labels()
    extra, extra_labels = cities.get_legend_handles_labels()
    ax.legend(handles + extra, labels + extra_labels, frameon=False,
              labelcolor=COLORS["faint"], fontsize=9, loc="upper left")
    return ax


def animate(state: GameState, recorder: GameRecorder, figsize=(13, 6.5),
            interval: int = 700):
    """The campaign as an inline animation, one frame a turn."""
    from matplotlib import animation

    fig = draw_map(state, recorder[0], figsize=figsize)

    def render(index):
        fig.clear()
        near, far = focus(state, recorder[index])
        grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.62), wspace=0.02)
        for i, (centre, label) in enumerate(((near, "the theatre"),
                                             (far, "the far side"))):
            ax = fig.add_subplot(grid[0, i])
            ax.set_facecolor(COLORS["background"])
            draw_globe(state, centre, recorder[index], ax=ax, label=label)
        panel = fig.add_subplot(grid[0, 2])
        panel.set_facecolor(COLORS["background"])
        panel.axis("off")
        _draw_panel(panel, state, recorder[index])
        fig.suptitle("Invasion: Earth — turn %d, %d of %d cities taken"
                     % (recorder[index].turn, recorder[index].taken,
                        len(state.geometry.urban)),
                     color=COLORS["text"], fontsize=13)
        return []

    return animation.FuncAnimation(fig, render, frames=len(recorder),
                                   interval=interval, blit=False)
