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
tests/test_ffw.py         49 tests, including the rulebook's worked examples
tools/                    data extraction, agent training, notebook generation
```

## Quick start

```bash
pip install numpy matplotlib ipywidgets nbformat jupyter
python -m unittest discover tests          # 49 tests, about 25 seconds
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

A full 40-week campaign runs in about four seconds, which is what makes
training practical.

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
| `HeuristicAgent` | weighted feature sum over every reachable system | ~4 s/game |
| `ScriptedAgent` | the same plus a historical opening | ~4 s/game |
| `LookaheadAgent` | rolls the game forward for each shortlisted destination | ~60 s/game |
| `NeuralAgent` | a self-play value network retunes the doctrine's aggression | ~5 s/game |

They share one weight vector of 18 named parameters, so a doctrine trained by
one can be handed to another.

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

**How much to trust the value network.** Not very much, at notebook scale, and
the code is built to make that visible. Every position in a game is labelled
with that game's final margin, so the independent sample size is the number of
*games*, not positions — twenty games is twenty samples however many hundred
rows the training matrix has. `evaluator_report` therefore scores the network on
held-out games and bootstraps the correlation **over games**:

```python
from ffw.training import train_value_network, evaluator_report
net, loss = train_value_network(games=20, seed=1)
print(evaluator_report(net, games=8))
# {'correlation': +0.27, 'ci': (-0.45, +0.73), 'spread': 0.09, ...}
```

Expect a positive point estimate with an interval straddling zero: the sign is
probably right, the magnitude is not measurable from a handful of games. That is
why `NeuralAgent` only nudges its doctrine with the network rather than choosing
moves with it.

Two earlier versions of this training loop were broken in ways the regression
loss did not reveal, which is the reason the diagnostic exists at all. The first
trained against a single fixed opponent, so every game ended the same way and
the network learned the mean of a nearly constant label — a superb loss and no
discrimination whatsoever. The second collected games from only one player's
seat and scored the other seat's games *backwards*, at a held-out correlation of
−0.49. In both cases the training loss looked excellent.

`tools/train_reference_agents.py` regenerates the trained agents committed under
`ffw/data/`.

## Writing your own agent

`ffw/agents/base.py` lists every decision the rules give a player, each with a
workable default. Override only what you want to change:

```python
from ffw.agents import HeuristicAgent

class CautiousImperial(HeuristicAgent):
    def wants_to_disengage(self, state, side, hex_id, engine):
        return True                      # never fight a fleet action
```

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
