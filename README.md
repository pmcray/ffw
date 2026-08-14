# Fifth Frontier War

A playable implementation of **Fifth Frontier War: Battles for the Spinward
Marches** (Game Designers' Workshop, 1981), Marc Miller's campaign game of the
Zhodani invasion of the Imperium, together with an interactive map, a human
interface, and several kinds of AI that can be trained and compared.

The rules, map, charts and counter sheets were read out of
`CT_G04_Traveller_Fifth_Frontier_War.pdf` in this repository.

```
FifthFrontierWar.ipynb    the notebook: watch, play, train, compare
ffw/                      the game
  hexmap.py               parsec geometry over Traveller CCRR hex numbers
  worlds/data/worlds.json 146 systems: starport, tech level, SDBs, battalions
  tables.py               every combat results table from the chart sheet
  oob.py                  counters and the two orders of battle
  state.py                worlds, squadrons, fleets, troops, admirals, control
  engine.py               setup and the four phases of the sequence of play
  features.py             the state encoding the value network sees
  training.py             cross-entropy doctrine search and self-play regression
  viz.py                  the star chart renderer and campaign recorder
  agents/                 random, heuristic, scripted, lookahead, neural, human
tests/test_ffw.py         87 tests, including the rulebook's worked examples
tools/                    data extraction, agent training, notebook generation
```

## Quick start

```bash
pip install numpy matplotlib ipywidgets nbformat jupyter
python -m unittest discover tests          # 87 tests, about 40 seconds
jupyter notebook FifthFrontierWar.ipynb
```

Or from Python:

```python
import ffw
from ffw.agents import ScriptedAgent
from ffw.viz import GameRecorder, draw_map

state = ffw.new_game(seed=7)
recorder = GameRecorder()
result = ffw.play(state,
                  ScriptedAgent('imperial', seed=1),
                  ScriptedAgent('zhodani', seed=2),
                  max_turns=40, on_turn=recorder)

print(result, state.victory_margin())
draw_map(state, recorder[-1])
```

A full 40-week campaign runs in about half a second, which is what makes
training practical.  It used to take four; the doctrine was asking "what is
near this system" by scanning all 146 worlds for every candidate destination of
every fleet, and answering that once instead of thousands of times made the
whole engine nine times faster without changing a single decision it takes.

## What is implemented

The whole sequence of play, in the order the rules give it.

**Movement.** Jump drives that cross whole hexes; fleets limited by their
slowest and thirstiest squadron; the refuelling table (gas giants, oceans,
starport capacity by type, bases, tanker squadrons); plotted movement written
four turns ahead for the Zhodani and five for everyone else, with an admiral's
planning factor replacing the fleet default; well-led fleets with planning
factor zero moving freely; independent scout squadrons; admirals travelling by
xboat network; plot aborts when reality diverges from the plan.

**Combat.** Space combat in rounds with the space CRT, multiple attacks above
48 factors, admirals' tactical factors, disengagement decided by the side with
the lower tactical ability first, and the optional squadron-quality column
shifts. System defence boats, active or passive, with the two SDB tables.
Surface bombing. Space-surface transfers. Surface combat on the troop CRT with
combat odds rounded in the defender's favour, tech level column shifts,
atmosphere die roll modifiers, armour and elite doubling, mercenaries losing
effectiveness past 50% losses, and psionic Zhodani Guards firing first.

**The rest.** Control and garrison requirements; airless worlds surrendering to
an unopposed warship; red zone interdiction lifting once the Zhodani arrive;
Ine Givar guerrillas going covert and overt and recovering strength; the
Zhodani secret base; Imperial reinforcement die rolls from turn 10 and the
colonial forces from turn 6; replacement points; victory points, the victory
table, automatic victory and the Zhodani unilateral armistice. Black globes and
jump troops are on by default and can be switched off in `state.options`.

## Where the data came from

The rules text (pages 1–21 of the PDF) is machine-readable. Everything else is
in scanned images, so:

- **The 146 world boxes** were read from the map (pages 27–30) at 260 dpi. Box
  background colour encodes atmosphere and the frame encodes travel zone, so
  those two were classified from pixel values rather than by eye, as were the
  water-availability dots.
- **Hex positions, gas giants and allegiances** come from the canonical
  Traveller Spinward Marches sector data. Fifth Frontier War numbers its hexes
  four columns and four rows off the canonical sector — Regina is canonical
  1910 and FFW 2314 — and every world the rules name by hex checks out under
  that offset, which is what `tests/test_ffw.py::test_rule_book_hexes` asserts.
- **The combat results tables** were transcribed from the chart sheet (page 22)
  and are tested against the rulebook's own worked examples.
- **The orders of battle** follow the two order-of-battle charts (pages 23–24)
  exactly for force levels. Individual counter values come from the counter
  sheets (pages 31–34); the scans do not support a reliable reading of all 720
  chits, so each category gets the mix of counter classes observed on the sheet
  expanded to the number of counters the chart calls for. Aggregate strengths
  match the printed game; a particular squadron may not match a particular chit.

Two typos in the rules are worth knowing about, since the data disagrees with
the text and the map settles it: the Imperial reinforcements entry hex is Jae
Tellona at **3218**, not 2218, and Efate is at **2109**, not the 2108 printed on
the Imperial order of battle chart. There is also an arithmetic slip in the
surface combat example, documented in the test that covers it.

## The AIs

Five agents, differing in *how* they decide:

| agent | approach | speed |
|---|---|---|
| `RandomAgent` | legal random moves — the control condition | fastest |
| `HeuristicAgent` | weighted feature sum over every reachable system | ~0.5 s/game |
| `ScriptedAgent` | the same plus a historical opening | ~0.5 s/game |
| `LookaheadAgent` | rolls the game forward for each shortlisted destination | ~25 s/game |
| `NeuralAgent` | a self-play value network retunes the doctrine's aggression | ~0.7 s/game |

Four of the five share one weight vector of 19 named parameters, so a doctrine
trained by one can be handed to another — which also means they are four
tunings of a single way of thinking.  `LookaheadAgent` is the outlier: it uses
the doctrine only to *propose* a shortlist and then decides by simulating each
candidate, so a match against it compares two architectures rather than two
weight vectors.

### The search agent, and what it cost to make it usable

`LookaheadAgent` used to take about fifteen seconds a turn, which put it out of
reach of any measurement worth having: a single paired comparison would have
taken most of a day. It now takes about two thirds of a second a turn, a
twenty-three-fold speedup, from three changes:

- **`GameState.clone`** instead of `copy.deepcopy`. Deep-copying a position
  walks the frozen counter classes, the route list and the whole log, none of
  which is ever written to. Cloning copies only what the engine mutates —
  worlds, squadrons, troops, admirals, fleets, pools — and shares the rest:
  0.3 ms against 5.8 ms, and a test plays a cloned position and a deep-copied
  one forward in lockstep to prove they are the same game.
- **Cached geometry.** Which worlds lie within *n* parsecs of a hex depends
  only on the map, and `state.Geometry` answers it once. This is what made the
  ordinary engine nine times faster too.
- **Common random numbers, and a budget.** Every candidate for a decision now
  rolls out on the *same* seed, so the comparison isolates the destination
  rather than the dice — the same trick `play_paired` uses on whole games — and
  only the few most important fleets each turn get searched at all.

Being affordable is not the same as being good, and the honest result is that
it is **not yet stronger than the doctrine it samples from**: −14.7 ± 6.5 VP
against `ScriptedAgent` over eight paired games. Two turns of war barely move
the scoreboard, so the raw margin at the leaf is mostly combat dice. The agent
accepts an `evaluator` for exactly that reason — a value network trained on
margin deltas, scoring the leaf instead of the scoreboard — but at the search
budget tested that did not rescue it either. `tools/lookahead_check.py` reruns
the comparison.

### Training

```python
from ffw.training import train_weights, train_value_network, tournament

weights, log = train_weights(side='zhodani', generations=6, population=10)
network, loss = train_value_network(games=8)
table, summary = tournament({...}, games=2)
```

`train_weights` runs the cross-entropy method over the doctrine weights: a
population plays, the best fraction is kept, the sampling distribution is
refitted. That learns *where fleets go*. `train_value_network` plays self-play
games, labels every position with the eventual victory margin, and fits a small
numpy MLP. `NeuralAgent` uses the network as a thermostat — consolidate when
winning, press when losing — which learns *how hard to push*. Both are pure
numpy; a trained agent is a small JSON file.

The value network is side-agnostic: `features.perspective` flips the state
vector so a single network serves both players, and `NeuralAgent` negates its
score for the Imperium. Games are collected from both seats for that reason.

**What the network is asked to predict.** There are two targets, and the choice
matters more than any other detail of the training loop.

- `target='final'` labels every position in a game with that game's final
  victory margin. Simple, and the reason the evaluator learned so little: the
  independent sample size is the number of *games*, not positions — twenty
  games is twenty samples however many hundred rows the training matrix has.
- `target='delta'` labels each position with the change in margin over the next
  six turns. Neighbouring positions now carry genuinely different labels, sixty
  games becomes thousands of samples, and the quantity predicted — *is this
  position about to improve* — is the one a thermostat and a search leaf
  actually want.

Trained on the same sixty games and each graded against its own target:

| target | held-out correlation | trivial baseline | skill over baseline |
|---|---|---|---|
| `final` | +0.815 [+0.75, +0.86] | +0.714 | **+0.101** |
| `delta` | +0.463 [+0.36, +0.57] | −0.142 | **+0.605** |

The delta network's raw correlation is *lower* and its real skill is six times
higher, which is the whole point. The feature vector contains the current
victory margin; late in a game that margin more or less *is* the final result,
so a network that echoes one column already scores +0.71 against the `final`
target without knowing anything. Against the `delta` target that same echo
scores −0.14, so every point of the delta network's +0.46 is earned.

Whether it plays better is a separate question, and the answer is **no**:
`NeuralAgent` driven by the delta network scored +1.6 ± 2.5 VP against the same
agent driven by the final-margin network over sixty paired games. Ten games had
said +10.2 ± 8.1, which is the usual lesson about this game's variance — a
promising result at one sample size and nothing at four times that.

A better evaluator that does not play better is worth understanding rather than
discarding. `NeuralAgent` uses the network as a thermostat: one scalar per turn
that shifts aggression, risk and concentration. That wiring was designed around
*am I winning*, and the delta network answers *am I gaining*, which is a
different question — the thermostat cannot exploit the extra accuracy because
it barely uses the number. The delta network's natural home is a search leaf,
where the question genuinely is "which of these positions is about to improve";
`LookaheadAgent` accepts it as an `evaluator` for that reason.
`tools/evaluator_trial.py` runs the whole comparison, and the network records
which target it was fitted to so the two cannot be silently swapped.

**How much to trust either of them.** Not very much at notebook scale, and the
code is built to make that visible. `evaluator_report` scores the network on
held-out games and bootstraps the correlation **over games**:

```python
from ffw.training import train_value_network, evaluator_report
net, loss = train_value_network(games=20, seed=1)
print(evaluator_report(net, games=8))
# {'correlation': +0.27, 'ci': (-0.45, +0.73), 'spread': 0.09, ...}
```

At notebook scale expect a positive point estimate with an interval straddling
zero. `evaluator_report` also reports the early game separately — before turn
twelve, the only place where an evaluator can show it knows something the
scoreboard has not already revealed. On the `final` target the network adds
essentially nothing there (+0.63 against a +0.58 baseline), which is why
`NeuralAgent` nudges its doctrine rather than choosing moves with it.

Two earlier versions of this training loop were broken in ways the regression
loss did not reveal, which is the reason the diagnostic exists at all. The first
trained against a single fixed opponent, so every game ended the same way and
the network learned the mean of a nearly constant label — a superb loss and no
discrimination whatsoever. The second collected games from only one player's
seat and scored the other seat's games *backwards*, at a held-out correlation of
−0.49. In both cases the training loss looked excellent.

### Training that can be shown to have worked

`train_weights` optimises one side against a *fixed* opponent, which rewards
doctrine that beats that opponent specifically. `train_league` co-evolves both
sides against a pool of each other's past champions, so the target moves as the
learner improves. It logs two numbers per generation and they answer different
questions:

- `score` — performance against the current opposing pool. **Not** comparable
  across generations: a flat score while the pool improves means the doctrine
  improved just as fast.
- `benchmark` — paired advantage over a fixed stock agent, played on both
  seats. This one **is** comparable, and it is the number to read.

```bash
python tools/train_at_scale.py          # league + value network + verification
python tools/train_at_scale.py verify   # just re-check the committed agents
```

Run at scale, the league's benchmark curves were `+23 +35 +32 +41 +41 +23`
(Imperial) and `+6 +0 +11 +11 +21 +21` (Zhodani) — upward, but with five paired
games per benchmark the wobble is noise, so the final verification is the number
to trust:

| side | advantage over stock | verdict |
|---|---|---|
| Zhodani | +25.4 ± 6.7 VP | better (3.8σ) |
| Imperial | +40.0 ± 9.3 VP | better (4.3σ) |

`train_at_scale.py` ends with a verification step: it plays the trained
doctrine against the stock one over paired games and reports the advantage with
a standard error and a verdict of better / worse / indistinguishable. That
verdict is stored in each agent's metadata, because training that cannot be
shown to have helped is not worth committing. `tools/train_one_side.py` and
`tools/train_reference_agents.py` remain for single-side and fixed-opponent
runs.

`tournament` plays **paired games**: each matchup runs twice on one seed with
the seats swapped, and the number reported is the difference between the two
halves. That matters because the seats are not equal — see the balance section
below — so raw margins rank the seat, not the player. Measured on this engine,
pairing cuts the standard deviation of a comparison from about 36 VP to 22;
it costs two games instead of one, so for fixed compute it buys roughly 15%
tighter error bars, and more importantly it makes the number mean "a outplayed
b" rather than "a drew the better side".

Each summary entry carries a `stderr`. A gap smaller than about two of them is
not a real difference: at one game per pairing that reliably separates `random`
from the rest, and does not resolve the doctrine agents against each other.

## Writing your own agent

`ffw/agents/base.py` lists every decision the rules give a player, each with a
workable default. Override only what you want to change:

```python
from ffw.agents import HeuristicAgent

class CautiousImperial(HeuristicAgent):
    def wants_to_disengage(self, state, side, hex_id, engine):
        return True                      # never fight a fleet action
```

## Is the game balanced?

Not evenly, and the code now measures it rather than guessing. `seat_bias` runs
the *same* doctrine on both sides, so whatever it reports is a property of the
game as implemented and not of the players:

```python
from ffw.training import seat_bias
seat_bias(games=8, max_turns=30)
# {'mean_margin': +151.8, 'stderr': 6.3, ...}   Zhodani favoured
```

Averaged over three campaigns, the victory points break down like this:

| | worlds taken | VP |
|---|---|---|
| Zhodani take Imperial worlds | 11.0 | 115.3 |
| Zhodani take neutral worlds | 11.0 | 38.2 |
| Imperium takes Zhodani worlds | 1.7 | 13.7 |
| Imperium takes neutral worlds | 2.0 | 6.3 |

Investigating that gap found two real bugs, both fixed and both now covered by
regression tests in `TestAmphibiousCapability`:

- **The Imperium could never embark a single troop factor.** Spare troops are
  offered to a fleet biggest-first, and the loader stopped at the head of the
  list instead of taking the largest unit that *fits*. A 500-factor army parked
  in front of a battle squadron's 20 factors of lift blocked every smaller unit
  behind it. Across a whole 40-turn war the Imperium put zero troops to sea.
- **The garrison rule was applied to worlds that do not need garrisoning.**
  Rule 5 binds on a world a player has *taken* — "control reverts to its
  original owner" — and a world cannot revert to the side that already holds
  it. Charging the owner 1% of its defence battalions locked 2296 factors of
  the Imperial army onto its own homeland; the Imperial troops had been
  deployed onto exactly the high-population worlds where that reservation bit
  hardest.

What is left after those fixes is largely the design. Imperial sealift is 814
troop factors against the Zhodani's 5009 — a six-to-one ratio that comes
straight from the order of battle, since the Consulate deploys six assault
carriers to the Imperium's one. The Zhodani are launching a surprise invasion
and are equipped for it. The armistice rule reads as the counterweight: from
turn 26 the Zhodani may end the war unilaterally at a cost of two victory
levels, which turns a typical +150 into a stalemate.

### The free worlds

Eighteen of the Marches' neutral worlds have no defence battalions and no
system defence boats. Fifty-five victory points sit on them, and five of them
are airless, which under rule 5 means an armed squadron can garrison them with
no troops at all. Left to itself the doctrine finished a war having claimed
**0.8 of the eighteen for the Imperium** and left nine unclaimed by anyone.

The reason turned out not to be the scoring. A fleet is an all-or-nothing
instrument, and no undefended backwater is ever worth diverting a battle fleet
for. `HeuristicAgent` now detaches a **picket** — one squadron on a spare fleet
marker, carrying a token troop unit — to go and sit on one. Getting that to
fire needed two things beyond the behaviour itself:

- **Fleet markers.** A player has fourteen, and the reorganisation step spent
  every one of them on the main effort, so the Imperium ended each turn with
  zero spare and could never form a picket at all. `reorganise_fleets` now
  takes a `reserve`.
- **Range.** The free worlds sit eight to twelve parsecs from where the
  Imperial fleets concentrate, far beyond one jump, so a picket has to be able
  to make its way there over several turns and to refuel when it arrives —
  which also means choosing a streamlined hull for the E-class ports.

Measured over thirty games, the behaviour moves the Imperial share from 0.8
worlds (1.5 VP) to 2.6 (7.0 VP), and drops the worlds nobody claims from 9.0 to
7.3. Its effect on the final margin is **+4.8 ± 5.7 VP over fifty paired
games** — the right sign, and smaller than the measurement can resolve. The
honest reading is that the points are real and the detour costs most of what it
earns; `tools/picket_check.py` reruns the whole comparison, reporting the
worlds claimed as well as the margin, because the direct tally is far less
noisy than a game result. Both halves can be switched off independently:
`pickets=False` stops the detachment, `free_world=0` stops free worlds pulling
on ordinary fleets.

## Known limits

- Fleets are capped at ten squadrons, which the rules do not require. Without a
  cap the AI merges everything into one blob whose jump number is set by its
  slowest squadron; the cap is doctrine, not a rule, and lives in
  `engine.MAX_FLEET_SIZE`.
- Initial deployment follows the order of battle's counts and required
  placements, but the discretionary placements are made by a simple rule rather
  than by the agent.
- Named colonial units are assigned to significant worlds of the right side
  rather than to the specific worlds printed on each counter, which the scans
  do not resolve.
- Xboat routes are the canonical Traveller network filtered to the mapped
  worlds, not a hand transcription of the green lines on the map.

## Credits

*Fifth Frontier War* and *Traveller* are the work of Marc W. Miller and Game
Designers' Workshop, 1981. This is an unofficial implementation for play and
research. Canonical sector data from travellermap.com.
