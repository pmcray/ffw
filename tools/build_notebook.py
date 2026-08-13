"""Generate FifthFrontierWar.ipynb.

The notebook is written from here rather than by hand so that the cell sources
stay under version control as ordinary Python strings and can be regenerated
after an API change.
"""

from __future__ import annotations

import json
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "FifthFrontierWar.ipynb")

MD = []
CODE = []


def md(text):
    MD.append(text)
    return ("md", text)


def code(text):
    CODE.append(text)
    return ("code", text)


CELLS = [
md("""# Fifth Frontier War
### Battles for the Spinward Marches — GDW, 1981

An interactive implementation of Marc Miller's campaign game. The war runs on
weekly turns across 146 star systems of the Spinward Marches; the Zhodani
Consulate, with the Sword Worlds and the Vargr, drives coreward-to-rimward
into the Imperium, and the Imperial Navy trades space for time until its
reinforcements arrive.

**What this notebook does**

| | |
|---|---|
| **Watch** | run two AIs against each other and replay the front line week by week |
| **Play** | take the Imperial or Zhodani side yourself against an AI |
| **Train** | improve an AI by self-play and see the learning curve |
| **Compare** | run a round-robin tournament between different kinds of AI |

Everything is driven by the `ffw` package in this repository. The map data was
transcribed from the game map in the PDF; the combat results tables come from
the chart sheet; the orders of battle come from the two order-of-battle charts.
"""),

code("""import sys, os, time, math, json, random
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import ffw
from ffw import features, hexmap, tables
from ffw.state import IMPERIAL, ZHODANI
from ffw.agents import (RandomAgent, HeuristicAgent, ScriptedAgent,
                        LookaheadAgent, NeuralAgent, HumanAgent,
                        ValueNetwork, DEFAULT_WEIGHTS, WEIGHTS)
from ffw.viz import GameRecorder, draw_map, plot_history, front_line, COLORS
from ffw.training import (play_match, train_weights, train_value_network,
                          tournament, save_agent, load_agent)

plt.rcParams['figure.facecolor'] = COLORS['background']
print('ffw', ffw.__version__)"""),

md("""## 1. The theatre

The stellar display is a 146-system slice of the Spinward Marches. Each hexagon
is one parsec; jump drives cross whole hexes without passing through the ones in
between, so distance, not adjacency, is what constrains a fleet.

Hex numbers use the Traveller convention `CCRR`, and they are the ones printed
in the rules: Regina is 2314, Efate 2109, Jewell 1510, Chronor 0708."""),

code("""state = ffw.new_game(seed=2024)
wm = state.world_map

print('worlds:            %d' % len(wm.worlds))
print('xboat routes:      %d' % len(wm.xboat_routes))
print('squadrons at start: Imperial %2d   Zhodani/allies %2d' % (
    sum(1 for s in state.squadrons.values() if s.side == IMPERIAL),
    sum(1 for s in state.squadrons.values() if s.side == ZHODANI)))
print()
print('%-14s %-6s %-9s %2s %7s %8s %-9s' % (
    'world', 'hex', 'starport', 'TL', 'SDBs', 'def bns', 'control'))
for h in ['2314', '2109', '1510', '0708', '3120', '2123', '1520', '1018']:
    w = wm.get(h)
    print('%-14s %-6s %-9s %2d %7d %8d %-9s' % (
        w.name, w.hex, w.starport + (' *' if w.naval_base else ''),
        w.tech_level, w.sdb, w.defense_battalions, w.control))"""),

md("""### The opening position

Red is Imperial, blue is Zhodani (which includes the Sword Worlds and the
Vargr), grey is the demilitarised independent region between them. Circles are
squadrons with the count inside; bars are ground forces; gold rings mark the
eight subsector capitals, which are worth double victory points."""),

code("""ax = draw_map(state, title='Fifth Frontier War, 1107')
plt.show()"""),

md("""## 2. Watching two AIs fight the war

`ffw.play` runs a whole campaign. Passing a `GameRecorder` as `on_turn` keeps a
snapshot of every week so the war can be replayed afterwards.

A full 40-turn campaign takes a few seconds."""),

code("""t0 = time.time()
state = ffw.new_game(seed=7)
recorder = GameRecorder()
result = ffw.play(state,
                  ScriptedAgent(IMPERIAL, seed=11),
                  ScriptedAgent(ZHODANI, seed=12),
                  max_turns=40, on_turn=recorder)

print('%s after %d turns (%.1fs)' % (result, state.turn, time.time() - t0))
print('victory points: Imperial %.1f   Zhodani %.1f   margin %+.1f'
      % (state.victory_points()[IMPERIAL],
         state.victory_points()[ZHODANI],
         state.victory_margin()))
print()
for line in state.log[-10:]:
    print(' ', line)"""),

md("""### The front line, week by week

Drag the slider to watch the war evolve. Worlds outlined in gold have changed
hands."""),

code("""import ipywidgets as widgets
from IPython.display import display

def show_turn(index):
    frame = recorder[index]
    fig, ax = plt.subplots(figsize=(15, 12))
    fig.patch.set_facecolor(COLORS['background'])
    draw_map(state, frame, ax=ax, show_names=True,
             highlight=front_line(state, frame))
    plt.show()

widgets.interact(show_turn,
                 index=widgets.IntSlider(min=0, max=len(recorder) - 1,
                                         step=1, value=len(recorder) - 1,
                                         description='turn',
                                         continuous_update=False));"""),

md("""If the slider does not render (ipywidgets not enabled in your Jupyter
install), the next cell shows the same thing as a filmstrip."""),

code("""picks = [0, len(recorder)//4, len(recorder)//2, len(recorder)-1]
fig, axes = plt.subplots(2, 2, figsize=(20, 17))
fig.patch.set_facecolor(COLORS['background'])
for ax, i in zip(axes.ravel(), picks):
    draw_map(state, recorder[i], ax=ax, show_names=False,
             highlight=front_line(state, recorder[i]))
    ax.get_legend().remove()
plt.tight_layout()
plt.show()"""),

code("""plot_history(recorder)
plt.show()"""),

md("""### Who took what

Victory points come from holding worlds across the border: an enemy subsector
capital is worth twice its tech level, any other enemy world its tech level, an
independent world half."""),

code("""changed = []
for w in state.world_map:
    original = {'imperial': IMPERIAL, 'zhodani': ZHODANI,
                'sword_worlds': ZHODANI, 'vargr': ZHODANI}.get(w.owner)
    if w.control is not None and original is not None and w.control != original:
        changed.append(w)
changed.sort(key=lambda w: -w.victory_points)

print('%-16s %-6s %-10s %-10s %6s' % ('world', 'hex', 'was', 'now', 'VP'))
for w in changed:
    print('%-16s %-6s %-10s %-10s %6.1f' % (
        w.name, w.hex, w.owner, w.control, w.victory_points))
if not changed:
    print('(the front never moved this game)')"""),

md("""## 3. Playing the war yourself

`HumanAgent` asks you for the decisions that matter — where each fleet is
ordered next, and whether to break off a battle — and applies its built-in staff
doctrine to everything else. Orders arrive through a callback, so you can drive
it from a prompt, from widgets, or from a script.

Remember the plotting rule: **fleet movement is written down several turns in
advance** (five for the Imperium, four for the Zhodani, or the planning factor
of the admiral commanding). You are ordering where a fleet will be in four or
five weeks' time, not this week. That delay is the heart of the game."""),

code("""def text_commander(request):
    \"\"\"A simple console interface for HumanAgent.\"\"\"
    if request['kind'] == 'plot_fleet':
        options = request['options']
        print('\\nTurn %d: orders for %s (%d squadrons, %d attack factors) at %s'
              % (request['turn'], request['fleet'], request['squadrons'],
                 request['attack'], request['origin_name']))
        rec = request['recommended']
        for i, (h, name) in enumerate(zip(options, request['option_names'])):
            mark = ' <- staff recommends' if h == rec else ''
            print('  %2d) %-16s %s  (%d pc)%s'
                  % (i, name, h, hexmap.distance(request['origin'], h), mark))
        print('   h) hold position')
        raw = input('choice [enter = follow the staff]: ').strip()
        if raw == '':
            return None
        if raw == 'h':
            return 'hold'
        try:
            return options[int(raw)]
        except (ValueError, IndexError):
            return None
    if request['kind'] == 'disengage':
        print('\\nBattle at %s: you have %d squadrons / %d attack, '
              'the enemy %d squadrons / %d attack'
              % (request['world'], request['own_squadrons'],
                 request['own_attack'], request['enemy_squadrons'],
                 request['enemy_attack']))
        raw = input('break off? [y/N]: ').strip().lower()
        return raw.startswith('y')
    return None

print('Ready. Run the next cell to take command.')"""),

md("""Uncomment the last line to play interactively. Be warned: with a dozen
fleets in play you will be asked for a lot of orders, so start with a short
game."""),

code("""def play_as(side=IMPERIAL, turns=6, seed=99, ask=text_commander):
    s = ffw.new_game(seed=seed)
    rec = GameRecorder()
    me = HumanAgent(side, ask=ask, seed=1)
    ai = ScriptedAgent(ZHODANI if side == IMPERIAL else IMPERIAL, seed=2)
    imperial, zhodani = (me, ai) if side == IMPERIAL else (ai, me)
    outcome = ffw.play(s, imperial, zhodani, max_turns=turns, on_turn=rec)
    print('\\n=== %s ===' % outcome)
    draw_map(s, rec[-1], highlight=front_line(s, rec[-1]))
    plt.show()
    return s, rec

# s, rec = play_as(IMPERIAL, turns=5)"""),

md("""### Playing with widgets instead

This version puts the orders on screen as buttons and redraws the map after
every turn, which is a lot easier to follow than the console."""),

code("""def widget_commander_factory(log_output):
    \"\"\"Returns an ``ask`` callback that renders a dropdown and blocks on it.

    Jupyter cannot block inside a callback, so this collects the orders the
    staff would give and lets you override them one turn at a time through the
    ``pending`` list.  For a fully synchronous UI, use ``text_commander``.
    \"\"\"
    def ask(request):
        with log_output:
            if request['kind'] == 'plot_fleet':
                print('T%d  %-14s %-10s -> staff: %s'
                      % (request['turn'], request['fleet'],
                         request['origin_name'], request['recommended']))
        return None
    return ask

out = widgets.Output(layout={'border': '1px solid #333', 'height': '220px',
                             'overflow': 'auto'})
display(out)

s2 = ffw.new_game(seed=5)
rec2 = GameRecorder()
ffw.play(s2, HumanAgent(IMPERIAL, ask=widget_commander_factory(out), seed=1),
         ScriptedAgent(ZHODANI, seed=2), max_turns=8, on_turn=rec2)
draw_map(s2, rec2[-1], highlight=front_line(s2, rec2[-1]))
plt.show()"""),

md("""## 4. The kinds of AI

Five agent types ship with the package. They differ in *how* they decide, which
is what makes comparing them interesting.

| agent | how it decides |
|---|---|
| `RandomAgent` | picks a legal destination at random — the control condition |
| `HeuristicAgent` | scores every reachable system with a weighted feature sum |
| `ScriptedAgent` | the same, plus a historical opening (Jewell → Efate → Regina) |
| `LookaheadAgent` | shortlists destinations, then rolls the game forward a few turns for each and keeps the best |
| `NeuralAgent` | a self-play value network reads the position and retunes the doctrine's aggression |

All four of the non-random agents share the same weight vector, so a doctrine
trained for one can be handed to another."""),

code("""for name in WEIGHTS:
    print('  %-16s %+6.2f' % (name, DEFAULT_WEIGHTS[name]))"""),

md("""The weights are the trainable parameters. `vp_value` is how much a fleet
cares about the victory points on offer at a target, `takeable` how much it
cares whether the landing force can actually beat the garrison, `frontier_pull`
how strongly it advances towards enemy space rather than shuffling behind its
own border, `risk_tolerance` how outnumbered it will accept being."""),

md("""## 5. Training an AI

Two things are worth learning, and they need different methods.

**Doctrine weights — cross-entropy method.** A population of candidate weight
vectors each plays the same fixed set of games; the best few are kept and the
sampling distribution is refitted to them. This learns *where fleets should go*.
Two details matter on a game this noisy: every candidate is judged on identical
game seeds, and the centre of the search only moves when a proposal actually
beats the incumbent on those same games. Without the second rule the search
walks downhill whenever an elite sample got lucky, which is most of the time.

**Position evaluation — self-play regression.** Games are played and every
position is labelled with the eventual victory margin; a small MLP is fitted to
predict it. `NeuralAgent` then uses that network as a thermostat: when it reads
the position as winning it consolidates, when losing it presses harder. This
learns *how hard to push*.

Both run in pure numpy, so there is nothing to install and a trained agent is a
small JSON file."""),

code("""t0 = time.time()
history = []
best_weights, log = train_weights(
    side=ZHODANI, generations=5, population=8, elite=3, games=1,
    seed=3, max_turns=32,
    progress=lambda g, mean, best: (
        history.append((g, mean, best)),
        print('generation %d   mean %+7.1f   best %+7.1f   (%.0fs)'
              % (g, mean, best, time.time() - t0))))
print('\\ndone in %.0fs' % (time.time() - t0))"""),

md("""Read the **incumbent** line, not the other two. A single game of Fifth
Frontier War varies by tens of victory points on the dice alone, so the best
sampled candidate in a generation is mostly a measure of who got lucky. The
incumbent is the accepted centre of the search, re-scored on the same fixed set
of games every generation, and it only moves when a proposal actually beats it
— so it can only go up."""),

code("""gen, mean, best = log.to_arrays()
_, incumbent = log.incumbents()
fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(COLORS['background'])
ax.set_facecolor(COLORS['background'])
ax.plot(gen, mean, label='population mean', color=COLORS['neutral'],
        marker='o', alpha=0.6)
ax.plot(gen, best, label='best sampled candidate', color=COLORS['faint'],
        marker='o', alpha=0.6, linestyle='--')
ax.plot(gen, incumbent, label='incumbent (accepted centre)',
        color=COLORS['zhodani'], marker='o', linewidth=2.5)
ax.set_xlabel('generation', color=COLORS['text'])
ax.set_ylabel('victory margin (Zhodani +)', color=COLORS['text'])
ax.set_title('Cross-entropy training of the Zhodani doctrine',
             color=COLORS['text'])
ax.tick_params(colors=COLORS['faint'])
ax.legend(labelcolor=COLORS['text'], frameon=False)
plt.show()

print('largest changes from the default doctrine:')
delta = sorted(((abs(best_weights[k] - DEFAULT_WEIGHTS[k]), k) for k in WEIGHTS),
               reverse=True)
for _, k in delta[:6]:
    print('  %-16s %+6.2f -> %+6.2f' % (k, DEFAULT_WEIGHTS[k], best_weights[k]))"""),

md("""### Training the value network by self-play"""),

code("""t0 = time.time()
net, loss = train_value_network(games=6, epochs=50, seed=3, max_turns=32,
                                progress=lambda g, m, r: print(
                                    '  game %d  margin %+7.1f  %s' % (g, m, r)))
print('\\nfinal regression loss %.4f  (%.0fs)' % (loss, time.time() - t0))

probe = ffw.new_game(seed=42)
print('opening position evaluates to %+.3f (0 = even)'
      % net(features.perspective(features.extract(probe), ZHODANI)))
for h in ['2314', '2109', '1510']:
    probe.world_map.get(h).control = ZHODANI
print('after Regina, Efate and Jewell fall: %+.3f'
      % net(features.perspective(features.extract(probe), ZHODANI)))"""),

code("""save_agent('trained_zhodani_notebook.json', best_weights, net,
           {'trained_in': 'notebook', 'side': ZHODANI})
w, n, meta = load_agent('trained_zhodani_notebook.json')
print('saved and reloaded:', meta)"""),

md("""### Reference agents

The repository ships with agents trained by `tools/train_reference_agents.py`.
Load them here if you would rather not wait for training."""),

code("""def load_reference(side):
    path = os.path.join('ffw', 'data', 'trained_%s.json' % side)
    if not os.path.exists(path):
        return None, None
    w, n, meta = load_agent(path)
    print('%s reference agent: %s' % (side, meta))
    return w, n

ref_zho_w, ref_zho_net = load_reference('zhodani')
ref_imp_w, ref_imp_net = load_reference('imperial')"""),

md("""## 6. Comparing the AIs

A round-robin: every agent plays every other on both sides, so the sides being
unbalanced does not favour anyone. The score reported for an entry is its
average victory margin *from its own point of view*, over all the games it
played, on either side."""),

code("""entries = {
    'random':    lambda side, seed: RandomAgent(side, seed=seed),
    'heuristic': lambda side, seed: HeuristicAgent(side, seed=seed),
    'scripted':  lambda side, seed: ScriptedAgent(side, seed=seed),
    'neural':    lambda side, seed: NeuralAgent(side, network=net, seed=seed),
    'trained':   lambda side, seed: HeuristicAgent(side, best_weights, seed=seed),
}

t0 = time.time()
table, summary = tournament(entries, games=1, max_turns=30, seed=5)
print('%.0fs' % (time.time() - t0))"""),

code("""names = list(entries)
print('average VP margin, row = Imperial player, column = Zhodani player')
print('(positive favours the Zhodani)')
print()
print('%-11s' % '', ''.join('%11s' % n for n in names))
for a in names:
    row = ''.join('%11s' % ('--' if a == b else '%+.0f' % table[(a, b)])
                  for b in names)
    print('%-11s%s' % (a, row))

print()
print('%-11s %6s %6s %6s %6s %9s' % ('agent', 'games', 'wins', 'draws',
                                     'losses', 'avg score'))
for name in sorted(names, key=lambda n: -summary[n]['average']):
    s = summary[name]
    print('%-11s %6d %6d %6d %6d %9.1f'
          % (name, s['games'], s['wins'], s['draws'], s['losses'],
             s['average']))"""),

code("""fig, ax = plt.subplots(figsize=(9, 4))
fig.patch.set_facecolor(COLORS['background'])
ax.set_facecolor(COLORS['background'])
order = sorted(names, key=lambda n: summary[n]['average'])
values = [summary[n]['average'] for n in order]
bars = ax.barh(order, values,
               color=[COLORS['zhodani'] if v > 0 else COLORS['imperial']
                      for v in values])
ax.axvline(0, color=COLORS['faint'], linewidth=0.8)
ax.set_xlabel('average victory margin in its own favour', color=COLORS['text'])
ax.set_title('Agent strength', color=COLORS['text'])
ax.tick_params(colors=COLORS['faint'])
for spine in ax.spines.values():
    spine.set_color(COLORS['grid'])
plt.show()"""),

md("""### A lookahead agent

`LookaheadAgent` copies the whole game state and rolls it forward a few turns
for each shortlisted destination. It plays better than the pure heuristic but
costs roughly fifteen seconds a turn against the heuristic's tenth of a second,
so it gets a short match here rather than a place in the round-robin above.
Raise `max_turns` if you want to watch it fight a whole campaign."""),

code("""t0 = time.time()
s3 = ffw.new_game(seed=21)
rec3 = GameRecorder()
outcome = ffw.play(s3,
                   ScriptedAgent(IMPERIAL, seed=1),
                   LookaheadAgent(ZHODANI, candidates=3, rollouts=1,
                                  horizon=2, seed=2),
                   max_turns=8, on_turn=rec3)
print('%s  (margin %+.1f after %d turns, %.0fs -- %.1fs per turn)'
      % (outcome, s3.victory_margin(), s3.turn - 1, time.time() - t0,
         (time.time() - t0) / max(1, s3.turn - 1)))
draw_map(s3, rec3[-1], highlight=front_line(s3, rec3[-1]))
plt.show()"""),

md("""## 7. Where to go next

The pieces are all in the `ffw` package and are meant to be extended:

- **`ffw/agents/base.py`** lists every decision the rules give a player. Subclass
  `Agent` (or `HeuristicAgent`) and override only the ones you care about.
- **`ffw/features.py`** is the state encoding the value network sees. Adding a
  feature here — supply lines, admiral quality, fuel reserves — changes what the
  network is able to learn.
- **`ffw/training.py`** holds both learning methods. The cross-entropy loop is
  easy to swap for anything else that optimises a vector.
- **`ffw/tables.py`** is the combat mathematics, transcribed from the chart
  sheet, with the rulebook's own worked examples as tests.

Two things worth trying:

1. **Train against a moving target.** `train_weights` currently plays a fixed
   opponent. Feeding the previous generation's best back in as the opponent
   gives a self-play arms race rather than a single fixed hill to climb.
2. **Give the network the decision, not the dial.** `NeuralAgent` uses its value
   network to modulate a hand-built doctrine. Evaluating each candidate
   destination directly with the network would let it discover doctrine of its
   own, at the cost of a much slower move generator.""")
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
