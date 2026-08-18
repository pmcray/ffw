"""Generate InvasionEarth.ipynb.

Written from here rather than by hand for the same reason ``build_notebook.py``
is: the cell sources stay under version control as ordinary Python strings and
can be regenerated after an API change.

A second notebook rather than a second half of the first one, because the two
games share a repository and almost nothing else -- a different map, a
different sequence of play, and a doctrine whose whole problem is getting to
the board at all.
"""

from __future__ import annotations

import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "InvasionEarth.ipynb")


def md(text):
    return ("md", text)


def code(text):
    return ("code", text)


CELLS = [
md("""# Invasion: Earth
### The assault on Terra — GDW, 1981

The second game in the *Traveller* series, and the end of the Solomani Rim War:
one player commands the Imperial invasion force, the other the defence of the
homeworld. The Imperium has to put four thousand combat factors of army onto a
planet it has to reach first; the Solomani own the ground already, and have
thirty-four system defence boat wings hidden in the oceans, planetary defences
that fire on anything landing within three hexes, and guerrillas that cannot be
bombarded at all.

**What this notebook does**

| | |
|---|---|
| **Look** | draw Terra as the sphere the rules describe, rather than the torn sheet the box prints |
| **Watch** | run the two doctrines against each other and replay the invasion turn by turn |
| **Read** | see where the doctrine's decisions come from, and what each weight does |
| **Compare** | measure the current doctrine against the one it replaced |

Everything is driven by the `ie` package in this repository."""),

code("""import sys, os, time, math, random
sys.path.insert(0, os.path.abspath('.'))

import matplotlib.pyplot as plt

from ie import hexmap, terra, oob
from ie.engine import new_game, play, Engine
from ie.agents import HeuristicAgent, ScriptedAgent, BeachheadAgent, RandomAgent
from ie.state import IMPERIAL, SOLOMANI
from ie.viz import GameRecorder, draw_map, draw_globe, plot_history, focus

plt.rcParams['figure.dpi'] = 110
print('%d cells, %d of them pentagons, about %.0f km across'
      % (hexmap.cell_count(), len(hexmap.PENTAGONS), hexmap.cell_width_km()))"""),

md("""## 1. The map is a sphere

The rulebook prints Terra "projected onto the twenty triangular sides of an
icosahedron", and then adds two rules whose only purpose is to undo the damage
the paper does:

> A hex divided between two or more of these triangles is considered to be a
> single hex for all purposes.
>
> The eastern and western edges of the map are considered to be adjacent, and a
> half hex on one side of the map has its other half on the other side.

Applied, those two sentences describe a **hex grid on a sphere**, which is
something a program can hold directly. So `ie/hexmap.py` builds a Goldberg
polyhedron — the dual of a subdivided icosahedron — rather than digitising a
scan of a folded sheet.

Every cell is a hexagon except exactly twelve pentagons at the icosahedron's
vertices, and that twelve is not a modelling choice: Euler's formula forbids
tiling a sphere with hexagons alone, which is *why* the printed map needs
seams. Find them in the picture below — they are the reason the rows of hexes
bend."""),

code("""state = new_game(seed=2024)
fig = draw_map(state)
fig.suptitle('Invasion: Earth — the opening position', color='#e8ecf2', fontsize=13)
plt.show()"""),

md("""Two hemispheres, so all 492 cells are on the page and adjacency across
the limb is where the eye expects it. The left globe is centred on wherever the
fighting is; at turn 1 there is none, so it centres on the Solomani mass.

The Imperial side of the board is entirely in the panel on the right: **every
Imperial unit starts in the out-system box**. That is the whole Imperial
problem, and it is a shipping problem before it is a fighting one — total lift
is 2875 combat factors against an army of 4040, and two of the four transport
squadrons are withdrawn on turn 2."""),

code("""cities = sorted(state.geometry.urban)
ports = sorted(state.geometry.starports)
print('%d urban hexes, %d starports, %d land hexes of %d'
      % (len(cities), len(ports), len(state.geometry.land), hexmap.cell_count()))
print()
print('the invasion has to garrison %d of the %d cities to end the war'
      % (len(cities) - 9, len(cities)))
print('Imperial army   %5d combat factors' % sum(
    c.factor * n for spec in (oob.IMPERIAL_TROOPS, oob.IMPERIAL_COLONIAL,
                              oob.IMPERIAL_MARINES, oob.IMPERIAL_MERCENARIES)
    for c, n in spec))
print('Imperial lift   %5d' % sum(c.capacity * n for c, n in oob.IMPERIAL_NAVAL))
print('Solomani army   %5d' % sum(c.factor * n for spec in (oob.SOLOMANI_TROOPS,
                                                            oob.SOLOMANI_PD)
                                  for c, n in spec))"""),

md("""## 2. Watching the invasion

Both sides are played by `HeuristicAgent`, one weight vector each. A campaign
of twenty-four turns takes a few seconds."""),

code("""t0 = time.time()
state = new_game(seed=7)
recorder = GameRecorder()
result = play(state, HeuristicAgent(IMPERIAL, seed=1),
              HeuristicAgent(SOLOMANI, seed=2),
              max_turns=24, on_turn=recorder)
print('%s after %d turns, %.1fs' % (result, state.turn - 1, time.time() - t0))
print('%d of %d cities garrisoned, %d victory points'
      % (recorder[-1].taken, len(state.geometry.urban), state.victory_points())) """),

md("""### The invasion, turn by turn

Drag the slider. The left globe follows the fighting, so it will jump when the
army moves."""),

code("""import ipywidgets as widgets
from IPython.display import display

def show_turn(index):
    fig = draw_map(state, recorder[index])
    plt.show()

widgets.interact(show_turn,
                 index=widgets.IntSlider(min=0, max=len(recorder) - 1,
                                         value=len(recorder) - 1,
                                         description='turn'))"""),

md("""If the slider does not render (ipywidgets not enabled in your Jupyter
install), the cell below draws the same thing at four points in the campaign."""),

code("""picks = [0, len(recorder)//3, 2*len(recorder)//3, len(recorder)-1]
for index in picks:
    fig = draw_map(state, recorder[index], figsize=(11, 5.5))
    plt.show()"""),

code("""plot_history(recorder)
plt.show()"""),

md("""Three curves and one story. The Imperial line climbs in steps — each step
is a convoy arriving — and then flattens when the lift runs out or the
transports are killed. The Solomani line falls, and then partly recovers at
each quarter boundary, because they draw **one replacement point per turn for
every ungarrisoned urban hex**: the defence funds itself from exactly the thing
the Imperium is trying to take away.

The dashed line is the scoreboard, and it is usually close to the floor. Why is
the subject of section 4."""),

md("""## 3. What the doctrine is thinking

`ie/agents/heuristic.py` plays both sides from one vector of named weights, the
same shape as `ffw`'s so a doctrine has something to transfer into."""),

code("""from ie.agents.heuristic import WEIGHTS
for name, value in WEIGHTS.items():
    print('%-20s %6.2f' % (name, value))"""),

md("""The Imperial half is built around three things the rules make true:

* **A hex is garrisoned by a zone of control, and lost to one.** A city is the
  Imperium's when it is in an Imperial zone of control and neither occupied by
  a Solomani unit nor in a Solomani zone of control. So the objective is never
  the city itself — it is a hex to stand on near several cities, with the local
  Solomani dead.
* **Grav units have ten movement points and a hex is 1100 km.** Both sides can
  cross a third of the planet in a turn. There is no safe beach, only distance
  from the mass.
* **The fleet is the only weapon that does not have to be shipped.** The
  surface bombardment table returns a *percentage of printed strength*, so
  thirty percent of a 500-factor field army is a hundred and fifty factors
  destroyed in a phase — more than the Solomani rebuild in a quarter.

The landing score is where most of that lands. Below it is evaluated over every
land hex of a fresh game, so you can see what the doctrine is actually looking
for."""),

code("""probe = new_game(seed=11)
agent = HeuristicAgent(IMPERIAL, seed=1)
field = (agent._threat_map(probe), agent._gun_map(probe), agent._close_map(probe))
scored = sorted(((agent._landing_score(probe, c, field), c)
                 for c in probe.geometry.land), reverse=True)
print('the ten best places to land, and the five worst that are not refused:')
for score, cell in scored[:10] + [(None, None)] + [
        s for s in scored[-5:] if s[0] > -1e8]:
    if cell is None:
        print('   ...')
        continue
    lat, lon = probe.geometry.lat_lon[cell]
    cities = sum(1 for c in probe.geometry.within(cell, 7)
                 if c in probe.geometry.urban)
    guns = field[1].get(cell, 0)
    print('   %6.1f  hex %3d  %-9s %5.1fN %6.1fE  %2d cities within 7, %2.0f gun factors'
          % (score, cell, probe.geometry.terrain[cell], lat, lon, cities, guns))"""),

md("""The refusals matter as much as the ranking. A hex scores `-1e9` — never
landed on at all — if it is ice, if a base could not be landed there (a base
"may not be landed in a tundra or ice hex", and an Imperial unit needs one
within five hexes to be in supply), or if there is not enough land within two
hexes to stand four thousand combat factors on at a thousand to the hex.

That last one is not hypothetical. The doctrine that preceded this one scored
cities within three hexes and charged thirty points a gun for batteries within
three hexes — and the batteries are *in* the cities, so no hex satisfied both.
What maximised the score was an empty island with three distant cities and no
way off it, and the invasion landed there and sat out the war."""),

md("""## 4. The rewrite, measured

`ie/agents/legacy.py` keeps that doctrine — both sides of it — so the claim
"the rewrite improved the invasion" can be a measurement in this engine rather
than a memory of a number from an older commit. `tools/ie_campaign.py` plays
whole campaigns and reports what became of both sides.

Four seeds a cell here so the notebook stays quick; the README quotes sixteen."""),

code("""from tools.ie_campaign import run

def compare(imperial, solomani, games=4, turns=24):
    _rows, summary, _levels = run(games, turns, imperial, solomani)
    return summary

print('%-24s %8s %8s %8s %8s %8s' % ('', 'cities', 'ashore', 'stranded',
                                     'squadns', 'Solomani'))
for imperial in ('beachhead', 'heuristic'):
    for solomani in ('beachhead', 'heuristic'):
        s = compare(imperial, solomani)
        print('%-24s %8.1f %8.0f %8.0f %8.1f %8.0f'
              % ('%s vs %s' % (imperial, solomani), s['taken'][0],
                 s['peak_ashore'][0], s['stranded'][0], s['fleet'][0],
                 s['solomani'][0]))"""),

md("""Read the rows and the invasion works: against the same defence the army
reaches Terra instead of a third of it reaching Terra, the fleet survives, and
the Solomani field army is cut roughly in half. Read the columns and the
defence works too — the Solomani half of the same rewrite fires the planetary
defences at the bombarding squadrons where they stand, at full factors rather
than the halved ones close orbit costs.

**And the cities column stays near zero.** That is the open problem, and it is
worth being precise about, because it is not the one you would guess. Of the
sixty-one cities, fifty-five to fifty-eight are ungarrisoned for the simplest
possible reason — no Imperial unit is standing within one hex of them — and
only one to four are blocked by a Solomani zone of control. The cell below
counts it in a finished game."""),

code("""geo = state.geometry
imperial_zoc = state.zone_of_control(IMPERIAL)
solomani_zoc = state.zone_of_control(SOLOMANI)
occupied = {u.location for u in state.surface.values()
            if u.side == SOLOMANI and u.carrier is None and not u.dead
            and isinstance(u.location, int)}
reasons = {'garrisoned': 0, 'no Imperial zone of control': 0,
           'occupied by the Solomani': 0, 'in a Solomani zone of control': 0}
for city in geo.urban:
    if city in state.garrisoned:
        reasons['garrisoned'] += 1
    elif city not in imperial_zoc:
        reasons['no Imperial zone of control'] += 1
    elif city in occupied:
        reasons['occupied by the Solomani'] += 1
    else:
        reasons['in a Solomani zone of control'] += 1
for reason, count in reasons.items():
    print('%-32s %2d' % (reason, count))"""),

md("""The army cannot spread out to cover them. A hex is only worth standing on
if what stands there survives what can reach it, and while thirty-six hundred
factors of grav-mobile Solomani troops are alive, that means stacks of about
five hundred — four or five stacks for the whole army, and a dozen cities
between them. Taking fifty-two needs the Solomani field army dead first, and at
the rate the exchange runs that is a longer war than the forty-eight turns the
engine allows."""),

md("""## 5. Writing your own doctrine

`ie/agents/base.py` lists every decision the rules give a player, each with a
workable default. Override only what you want to change."""),

code("""class NeverBombards(HeuristicAgent):
    \"\"\"What happens if the fleet keeps its guns holstered?\"\"\"
    def missions(self, state, side, engine):
        return {}

quiet = new_game(seed=31)
result = play(quiet, NeverBombards(IMPERIAL, seed=1),
              HeuristicAgent(SOLOMANI, seed=2), max_turns=18)
loud = new_game(seed=31)
result2 = play(loud, HeuristicAgent(IMPERIAL, seed=1),
               HeuristicAgent(SOLOMANI, seed=2), max_turns=18)
for name, s in (('no bombardment', quiet), ('the doctrine', loud)):
    print('%-16s %4d Solomani factors left, %2d cities taken, %2d squadrons left'
          % (name,
             sum(u.current for u in s.surface.values()
                 if u.side == SOLOMANI and not u.dead and not u.cls.guerrilla),
             len(s.geometry.urban) - len(s.solomani_urban()),
             len([u for u in s.naval.values() if u.side == IMPERIAL])))"""),

md("""## 6. Where to go next

- **A human interface.** `ffw` has one (`ffw/agents/human.py`) and this game
  does not, so the invasion can be watched but not yet played. The decisions a
  human would want are already enumerated in `ie/agents/base.py`.
- **Training.** `ffw/training.py`'s cross-entropy doctrine search optimises a
  named weight vector against a fixed or co-evolving opponent, and this game's
  weights have the same shape. `strategy/adapter.py` is the layer that would
  let it drive either game.
- **The garrison problem.** Section 4 is the honest state of it: the invasion
  works and the conquest does not. Anything that kills the Solomani field army
  faster — better odds selection, bombardment aimed at the units about to
  fight, a landing that splits the defence between two theatres — is aimed at
  the right target.
- **The terrain.** Terra's coastlines are authored as data (longitude spans per
  latitude band, in `ie/terra.py`) rather than sampled off the scan, so they
  can be read and argued with. They are recognisable rather than accurate, and
  the 61 urban hexes are placed by ranking land hexes against named regions.""")
]


def build():
    nb = nbf.v4.new_notebook()
    cells = []
    for kind, source in CELLS:
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(source))
        else:
            cells.append(nbf.v4.new_code_cell(source))
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        nbf.write(nb, fh)
    print("wrote", OUT, "with", len(cells), "cells")


if __name__ == "__main__":
    build()
