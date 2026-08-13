"""Drawing the Spinward Marches.

``draw_map`` renders the stellar display as a matplotlib figure: one hexagon
per system, coloured by who controls it, with fleet and army markers on top.
``GameRecorder`` keeps a light-weight snapshot of every turn so a finished
campaign can be replayed with a slider, and ``animate`` turns the snapshots
into an inline animation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.lines import Line2D

from . import hexmap
from .state import GameState, IMPERIAL, ZHODANI

# a dark "star chart" palette that stays readable in both notebook themes
COLORS = {
    "background": "#11151c",
    "grid": "#2c3440",
    "imperial": "#d94f4f",
    "zhodani": "#4f8fd9",
    "neutral": "#8a8f98",
    "sword_worlds": "#4fbf7f",
    "vargr": "#d99a4f",
    "text": "#e8ecf2",
    "faint": "#7f8894",
    "route": "#3f6b4f",
}

SIDE_COLOR = {IMPERIAL: COLORS["imperial"], ZHODANI: COLORS["zhodani"],
              None: COLORS["neutral"]}


@dataclass
class Snapshot:
    """Everything the map renderer needs for one turn."""
    turn: int
    control: dict
    fleets: list
    troops: dict
    margin: float
    victory: str
    log: list = field(default_factory=list)


def snapshot(state: GameState) -> Snapshot:
    control = {w.hex: w.control for w in state.world_map}
    fleets = []
    for fleet in state.fleets.values():
        if not fleet.active or fleet.location not in state.world_map.worlds:
            continue
        attack = sum(state.squadrons[u].attack for u in fleet.squadrons
                     if u in state.squadrons)
        fleets.append({"hex": fleet.location, "side": fleet.side,
                       "name": fleet.name, "squadrons": len(fleet.squadrons),
                       "attack": attack})
    troops: dict = {}
    for unit in state.troops.values():
        if unit.aboard is not None or unit.losses >= 100:
            continue
        if unit.location not in state.world_map.worlds:
            continue
        key = (unit.location, unit.side)
        troops[key] = troops.get(key, 0) + unit.current
    return Snapshot(turn=state.turn, control=control, fleets=fleets,
                    troops=troops, margin=state.victory_margin(),
                    victory=state.victory_level(), log=list(state.log[-6:]))


class GameRecorder:
    """Collects one snapshot per turn; pass ``recorder`` as ``on_turn``."""

    def __init__(self, keep_log: bool = True):
        self.frames: list[Snapshot] = []
        self.keep_log = keep_log

    def __call__(self, state: GameState) -> None:
        self.frames.append(snapshot(state))

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index) -> Snapshot:
        return self.frames[index]


# --------------------------------------------------------------------------
def _layout(world_map):
    positions = {}
    for hex_id in world_map.worlds:
        col, row = hexmap.parse(hex_id)
        x = 1.5 * col
        y = math.sqrt(3) * (row + (0.0 if col % 2 else 0.5))
        positions[hex_id] = (x, -y)
    return positions


def draw_map(state: GameState, frame: Snapshot | None = None,
             ax=None, figsize=(15, 12), show_names: bool = True,
             show_routes: bool = True, title: str | None = None,
             highlight: set | None = None):
    """Render the stellar display.  ``frame`` replays a recorded turn."""
    world_map = state.world_map
    control = frame.control if frame else {w.hex: w.control for w in world_map}
    fleets = frame.fleets if frame else snapshot(state).fleets
    troops = frame.troops if frame else snapshot(state).troops
    turn = frame.turn if frame else state.turn
    margin = frame.margin if frame else state.victory_margin()
    victory = frame.victory if frame else state.victory_level()

    positions = _layout(world_map)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    ax.set_aspect("equal")
    ax.axis("off")

    if show_routes:
        for a, b in world_map.xboat_routes:
            if a in positions and b in positions:
                xa, ya = positions[a]
                xb, yb = positions[b]
                ax.plot([xa, xb], [ya, yb], color=COLORS["route"],
                        linewidth=1.0, zorder=1, alpha=0.7)

    radius = 1.0
    for hex_id, world in world_map.worlds.items():
        x, y = positions[hex_id]
        side = control.get(hex_id)
        face = SIDE_COLOR.get(side, COLORS["neutral"])
        original = {"imperial": IMPERIAL, "zhodani": ZHODANI,
                    "sword_worlds": ZHODANI, "vargr": ZHODANI}.get(world.owner)
        conquered = side is not None and original is not None and side != original
        hexagon = RegularPolygon((x, y), numVertices=6, radius=radius,
                                 orientation=math.pi / 6, facecolor=face,
                                 edgecolor=(COLORS["text"] if conquered
                                            else COLORS["grid"]),
                                 linewidth=1.8 if conquered else 0.7,
                                 alpha=0.30 if side else 0.16, zorder=2)
        ax.add_patch(hexagon)
        if highlight and hex_id in highlight:
            ax.add_patch(RegularPolygon((x, y), numVertices=6, radius=radius,
                                        orientation=math.pi / 6, facecolor="none",
                                        edgecolor="#f5d76e", linewidth=2.4,
                                        zorder=6))
        star = Circle((x, y - 0.15), 0.22,
                      facecolor="#f2f2f2" if world.starport in "ABCD" else "#9aa3ad",
                      edgecolor="none", zorder=4)
        ax.add_patch(star)
        if world.subsector_capital:
            ax.add_patch(Circle((x, y - 0.15), 0.36, facecolor="none",
                                edgecolor="#f5d76e", linewidth=1.2, zorder=4))
        if show_names:
            ax.text(x, y + 0.66, world.name[:12], ha="center", va="center",
                    fontsize=5.4, color=COLORS["text"], zorder=5)
        if world.naval_base or world.military_base:
            ax.text(x - 0.62, y + 0.30, "*", fontsize=8,
                    color=COLORS["faint"], zorder=5)
        if world.gas_giant:
            ax.add_patch(Circle((x + 0.55, y + 0.38), 0.09,
                                facecolor=COLORS["faint"], edgecolor="none",
                                zorder=5))

    # forces
    strength: dict = {}
    for fleet in fleets:
        key = (fleet["hex"], fleet["side"])
        entry = strength.setdefault(key, {"squadrons": 0, "attack": 0})
        entry["squadrons"] += fleet["squadrons"]
        entry["attack"] += fleet["attack"]
    for (hex_id, side), entry in strength.items():
        if hex_id not in positions:
            continue
        x, y = positions[hex_id]
        dx = -0.42 if side == IMPERIAL else 0.42
        size = 0.14 + 0.035 * math.sqrt(entry["squadrons"])
        ax.add_patch(Circle((x + dx, y - 0.55), size,
                            facecolor=SIDE_COLOR[side], edgecolor=COLORS["text"],
                            linewidth=0.5, zorder=7))
        ax.text(x + dx, y - 0.55, str(entry["squadrons"]), ha="center",
                va="center", fontsize=4.6, color="#101010", zorder=8)
    for (hex_id, side), factors in troops.items():
        if hex_id not in positions or factors <= 0:
            continue
        x, y = positions[hex_id]
        dx = -0.42 if side == IMPERIAL else 0.42
        width = 0.20 + 0.05 * math.log10(max(10, factors))
        ax.add_patch(plt.Rectangle((x + dx - width / 2, y + 0.24),
                                   width, 0.18,
                                   facecolor=SIDE_COLOR[side],
                                   edgecolor=COLORS["text"], linewidth=0.4,
                                   zorder=7))

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 2, max(xs) + 2)
    ax.set_ylim(min(ys) - 2, max(ys) + 2)

    heading = title or "Fifth Frontier War"
    ax.set_title("%s  --  turn %d   VP margin %+.1f   (%s)"
                 % (heading, turn, margin, victory),
                 color=COLORS["text"], fontsize=13, pad=14)
    legend = [
        Line2D([], [], marker="h", linestyle="none", markersize=10,
               markerfacecolor=COLORS["imperial"], markeredgecolor="none",
               label="Imperial control"),
        Line2D([], [], marker="h", linestyle="none", markersize=10,
               markerfacecolor=COLORS["zhodani"], markeredgecolor="none",
               label="Zhodani control"),
        Line2D([], [], marker="h", linestyle="none", markersize=10,
               markerfacecolor=COLORS["neutral"], markeredgecolor="none",
               label="uncontrolled"),
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=COLORS["text"], markeredgecolor="none",
               label="squadrons (count inside)"),
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               markerfacecolor=COLORS["text"], markeredgecolor="none",
               label="ground forces"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=8,
              labelcolor=COLORS["text"], ncol=5,
              bbox_to_anchor=(0.0, -0.02))
    return ax


# --------------------------------------------------------------------------
def plot_history(recorder: GameRecorder, ax=None, figsize=(10, 4)):
    """Victory-point margin over the course of the campaign."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    turns = [f.turn for f in recorder.frames]
    margins = [f.margin for f in recorder.frames]
    ax.plot(turns, margins, color=COLORS["zhodani"], linewidth=2)
    ax.axhline(0, color=COLORS["faint"], linewidth=0.8)
    ax.axhspan(-50, 50, color=COLORS["faint"], alpha=0.15)
    ax.set_xlabel("turn (weeks)", color=COLORS["text"])
    ax.set_ylabel("VP margin (Zhodani +)", color=COLORS["text"])
    ax.tick_params(colors=COLORS["faint"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["grid"])
    ax.set_title("Course of the war", color=COLORS["text"])
    return ax


def animate(state: GameState, recorder: GameRecorder, figsize=(14, 11),
            interval: int = 700, step: int = 1):
    """Return a matplotlib animation of the recorded campaign."""
    from matplotlib import animation
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(COLORS["background"])
    frames = recorder.frames[::step]

    def render(index):
        ax.clear()
        draw_map(state, frames[index], ax=ax, show_names=False)
        return ax,

    return animation.FuncAnimation(fig, render, frames=len(frames),
                                   interval=interval, blit=False,
                                   repeat=False)


def front_line(state: GameState, frame: Snapshot | None = None) -> set:
    """Worlds that changed hands relative to their original owner."""
    control = frame.control if frame else {w.hex: w.control for w in state.world_map}
    out = set()
    for world in state.world_map:
        original = {"imperial": IMPERIAL, "zhodani": ZHODANI,
                    "sword_worlds": ZHODANI, "vargr": ZHODANI}.get(world.owner)
        if original is not None and control.get(world.hex) not in (None, original):
            out.add(world.hex)
    return out
