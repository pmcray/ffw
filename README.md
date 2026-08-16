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
  llm.py                  Claude proposes doctrine edits; the engine grades them
  viz.py                  the star chart renderer and campaign recorder
  agents/                 random, heuristic, scripted, lookahead, neural,
                          doctrine (written rules), human
tests/test_ffw.py         148 tests, including the rulebook's worked examples
tools/                    data extraction, agent training, notebook generation
```

## Quick start

```bash
pip install numpy matplotlib ipywidgets nbformat jupyter
python -m unittest discover tests          # 148 tests, about 50 seconds
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
| `DoctrineAgent` | a written rule set — conjunctions, not coefficients | ~0.8 s/game |

Three of the six share one weight vector of 19 named parameters, so a doctrine
trained by one can be handed to another — which also means they are three
tunings of a single way of thinking.  The other three differ in kind:
`LookaheadAgent` uses the doctrine only to *propose* a shortlist and then
decides by simulating each candidate, and `DoctrineAgent` throws the weighted
sum out entirely and decides by matching named rules.

### Doctrine written in words

`DoctrineAgent` is the odd one out in a different direction. Its decisions come
from an ordered list of **rules** — each a conjunction of named conditions and
one effect — rather than a weighted sum:

```json
{"name": "occupy free worlds within reach",
 "when": ["claimable", "carrying>=1", "distance<=3"],
 "then": {"prefer": 2.4},
 "rationale": "Eighteen neutrals have no defences at all and carry 55 VP."}
```

The reason to have it is not style. **A weight vector cannot express a
conjunction.** Whatever coefficient the linear doctrine puts on `undefended` it
pays at every distance, carrying troops or empty — no vector of per-feature
weights says "undefended *and* within two jumps *and* I have troops aboard".
That sentence is one obvious instruction to a human commander and it is outside
the search space of every numeric trainer here. A test pins the difference down:
under a rule set the same world scores differently depending on the
*combination* of facts, and identically under any single weight.

Two more tests keep the claim honest. An empty rule set must score every
destination identically — if any term of the weighted sum were leaking through,
it wouldn't. And the condition vocabulary published in the prompt is generated
from the parser, so the two cannot drift apart.

It plays at the level of the doctrine it replaces, which is what makes the
comparison fair rather than flattering:

| matchup | over 30 paired games |
|---|---|
| doctrine vs `ScriptedAgent` | +7.0 ± 4.6 VP (indistinguishable) |
| doctrine vs `HeuristicAgent` | +0.3 ± 4.4 VP (indistinguishable) |

### The proposal loop: doctrine proposed in words, graded by the engine

A search space of *behaviours* is only useful if something can search it.
`ffw/llm.py` is that something. Each generation Claude is shown how the current
doctrine actually performed — where the victory points went, how many of the 18
undefended neutrals finished claimed by nobody, which rules never fired once in
a whole war, and what every previous proposal measured — and proposes edits in
the rule vocabulary. The engine then plays each candidate against the incumbent
over paired games and reports an advantage with a standard error.

```bash
export ANTHROPIC_API_KEY=...            # or: ant auth login
pip install anthropic
python tools/evolve_doctrine.py --generations 5 --proposals 4 --games 12
python tools/evolve_doctrine.py --dry-run     # same loop, scripted proposals
```

The division of labour is the point. The model supplies imagination over a space
numeric search cannot reach; the engine supplies the one thing a model cannot
supply about its own ideas, which is whether they are true. **This is the loop
that would have found the free-world problem without a person noticing it** — it
is handed the "claimed by nobody" line every single generation.

Three things stop it fooling itself, each one a lesson this project already paid
for:

- **Unparseable proposals are rejected before a game is played**, and the error
  goes into the next briefing verbatim — the model is told which term it
  invented and what the vocabulary actually contains.
- **A winner must clear `--accept-sigma` standard errors.** At zero the loop
  accepts whichever candidate the seeds favoured and calls a random walk
  progress; at two nothing clears the bar at any affordable sample size. One is
  the compromise, and the verdict prints beside it so a weak acceptance is
  visible as one.
- **A winner must then win again on fresh seeds.** Best-of-four on one set of
  games is partly selected for fitting those games — precisely the trap that
  made the evaluator comparison read +10.2 at ten games and +1.6 at sixty.

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

### Measuring precisely enough to get an answer

A paired game costs about 1.6 s and the standard error of a comparison falls as
1/√games, so twenty paired games leaves ±4.7 VP — wider than most effects worth
measuring here. Every "indistinguishable" verdict in this project came from
sample sizes chosen when the engine was nine times slower, and nobody had spent
the speedup.

Games differ only by seed and share no state, so they parallelise perfectly. The
only obstacle was that every agent factory in the codebase was a lambda, and a
lambda cannot be pickled. `AgentSpec` is a factory made of plain data instead:

```python
from ffw.training import AgentSpec, evaluate_paired
result = evaluate_paired(AgentSpec("doctrine"), AgentSpec("scripted"), games=120)
# {'advantage': +6.5, 'stderr': 3.6, 'verdict': 'indistinguishable', ...}
```

3.8× on four cores, and **a test asserts parallel and serial return identical
lists** — each game is seeded independently and results come back in seed order.
A lambda still works and simply runs in this process; being unpicklable is a
reason to be slower, not to fail. `FFW_WORKERS=1` pins it to one core.

Two things were fixed on the way. `summarise` is now the single place that turns
advantages into a verdict — that arithmetic had been copy-pasted into four tools,
three too many for a number every conclusion rests on. And `tournament` seeded
its pairings from `hash((a, b))`, which **Python randomises per process**: the
one function whose entire job is comparable numbers had never been reproducible
across runs. It uses a stable CRC now, with a test that spawns two interpreters
and compares.

### The first tournament with error bars worth reading

40 paired games per pairing, five agents, about eight minutes:

| agent | average | stderr | record |
|---|---|---|---|
| `HeuristicAgent` | +18.8 | ±2.7 | 104–56 |
| `DoctrineAgent` | +16.5 | ±3.0 | 103–57 |
| `NeuralAgent` | +13.4 | ±2.8 | 92–68 |
| `ScriptedAgent` | +12.2 | ±2.9 | 101–59 |
| `RandomAgent` | −60.8 | ±1.6 | 0–160 |

The head-to-head numbers say more than the ranking:

- **`ScriptedAgent`'s historical opening costs about 9 VP** (`heuristic` beats it
  +9.4). It has been the "stock" baseline every training run was measured
  against, and scripting the drive on Jewell–Efate–Regina is worse than letting
  the doctrine choose.
- **The value network makes play worse, not better** (`neural` loses to
  `heuristic` by 4.9). That is consistent with everything else measured about it:
  it predicts adequately and steers badly.
- **`DoctrineAgent` is level with `HeuristicAgent`** (+0.1 over 40 paired games),
  which is what makes the rule-based architecture a fair comparison rather than a
  handicapped one.

### A behaviour that turned out to be side-specific

Re-running the picket question at 120 paired games — impossible before, ~2
minutes now — showed the effect is asymmetric, and that the earlier verdict was
noise in both directions:

| | Imperial | Zhodani |
|---|---|---|
| first 120 games | +6.5 ± 3.6 | **−9.2 ± 4.0** ("worse") |
| fresh 120 games | +3.4 ± 3.2 | −3.6 ± 3.9 (indistinguishable) |
| **pooled, 240** | **+5.0 ± 2.4** | **−6.4 ± 2.8** |

The first Zhodani block cleared 2σ and the confirmation block did not — the
regression-to-the-mean lesson arriving on schedule, which is why the pooled
figure is the one quoted. The reading is plain: the Imperium has nothing better
for a spare escort to do, while the Zhodani are already collecting eight free
worlds in the course of an advance their 6:1 sealift advantage is built around,
so detaching a squadron costs the main effort more than the marginal world
returns.

The default is now side-aware, and verified on a **third** untouched seed block
at **+3.7 ± 2.1** against the old always-on behaviour — positive in all three
blocks, 1.8σ rather than decisive. `pickets=True/False` still overrides it.

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

| side | vs `ScriptedAgent` | vs `HeuristicAgent` | verdict |
|---|---|---|---|
| Zhodani | +11.1 ± 1.9 VP | +2.4 ± 2.0 VP | **indistinguishable** |
| Imperial | +20.0 ± 2.2 VP | +14.5 ± 2.0 VP | better (7.3σ) |

Those are the corrected figures, and the correction matters. The first run of
this table read `+25.4 ± 6.7` and `+40.0 ± 9.3`, measured over ten paired games
against `ScriptedAgent`. Two things were wrong with it. Ten games is not enough
to state a number to one decimal place — parallel evaluation made 120 games
affordable and the error bars fell by a factor of three. And `ScriptedAgent`'s
historical opening is itself worth about five victory points against plain
`HeuristicAgent`, so "beats stock" was partly a measurement of a self-inflicted
handicap. Re-run against an opponent that is not holding a brick, **the trained
Zhodani doctrine is no better than the untrained one**; the trained Imperial
doctrine is genuinely stronger, by about a third less than first reported.
`tools/rebaseline.py` re-runs the whole comparison and writes
`ffw/data/rebaseline.json`.

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

### The turn the game actually ends

The rulebook gives no turn limit. Play continues "until a player achieves an
automatic victory or until an armistice occurs", and rule 8 prices that
armistice precisely:

> If the Zhodani player unilaterally declares an armistice on or between turns
> 26 to 51, the level of victory is shifted two levels in favor of the Imperial
> player. If the Zhodani player unilaterally declares an armistice on or after
> turn 52, the level of victory is shifted one level in favor of the Imperial
> player.

Every measurement in this project had been run to turn 30, 40 or 45 for speed.
That is *before the game's own ending condition can fire* — so the armistice
never happened in any of them, and the doctrine's handling of it was never
graded. Left unexamined, it declared at turn 40 whenever it was ahead, paying
two victory levels for nothing. Waiting for the turn-52 boundary costs one.

Simply moving that boundary is worth **+0.90 ± 0.08 victory levels over sixty
games** — the single largest effect measured anywhere in this project, and it
comes from reading the rulebook rather than from any training. The outcome
distribution moves from `stalemate ×47, Zhodani marginal ×12, Imperial marginal
×1` to `Zhodani marginal ×38, Zhodani major ×14, stalemate ×7, Imperial marginal
×1`. The victory *margin* barely moves: +173 to +179.

That gap between margin and level is the point, and it is why the project now
keeps two measures:

- **`evaluate_paired`** scores by victory-point margin. It stays the right tool
  for *comparing play*, because it is continuous and resolves differences the
  nine-step victory table rounds away.
- **`outcome_series`** scores by victory level, and defaults to `max_turns=60`
  so the boundary that decides the game is actually reachable. Use it when the
  question is who won rather than by how much.

A doctrine tuned only on margin optimises a quantity that does not settle the
game. `TestArmistice` pins the rule down, including the rulebook's own worked
example: turn 34, margin 127, a Zhodani major victory shifted two levels to a
stalemate.

### What a rules audit turned up

Rule 6 was implemented in outline and wrong in three specifics, all found by
reading the booklet against the code rather than by any failing test:

- **Fleet markers never arrived.** The reinforcements table's third column is
  fleet markers; the code read that column, spawned the admiral and special
  counter that accompany a fleet, and never released the marker itself.
  Meanwhile all fourteen were spendable from turn 1, so rule 3's "a player may
  have only a limited number of fleets in play, as determined by his order of
  battle" had no force at all, and the Imperium opened with ten markers it
  should have had to wait for. `Fleet.available` now separates *the player owns
  this marker* from `Fleet.active`, *it is on the map*.
- **The black globes were in the ordinary draw pile.** Rule 9 sets the four
  6-2-8 battle squadrons aside — "they are not selected at random for initial
  forces or reinforcements" — and releases them on the Imperial player's first
  roll of 6, in place of that roll's usual units. They are now a group of their
  own.
- **The warrant could never reach anyone.** It is the only way the Imperial
  navy may land troops on an interdicted world, and the agent code correctly
  refused to land without it — but the warrant was a counter in a pool nothing
  ever drew from, so no admiral could ever hold one. It is one counter in rule
  6's special group, so it now arrives on a die roll partway through the war,
  and sets its holder's precedence to 0.

Restricting the Imperium's opening fleet markers sounds like a balance change
and turned out not to be one: seat bias over 120 paired games moved from
+173.4 ± 3.3 to +171.2 ± 3.5, a difference of 2 ± 5. Correctness, not balance.
Fourteen tests in `TestReinforcementSchedule` hold the schedule to the booklet.

One number in the audit is not quoted rule. The opening fleet allotment lives
on the order of battle charts, which are printed on the player aid rather than
in the rules booklet, and so are not in the PDF this project reads. What the
booklet does settle is that the allotment is *smaller* than the counter mix —
otherwise the turn-2 draw of three markers would mean nothing. The figures in
`oob.INITIAL_FLEETS` are the project's best reading, and are marked as such.

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
7.3. Its effect on the final margin is **+5.2 ± 1.4 VP over 240 paired games**
— better at 3.7σ. That figure took three attempts to pin down, and the first
two are worth recording:

- At fifty paired games it read **+4.8 ± 5.7**: the right sign, and smaller
  than the measurement could resolve. The honest reading at the time was that
  the points were real and the detour cost most of what it earned. Parallel
  evaluation made 240 games affordable, and the same effect is now four
  standard errors clear of zero. Nothing changed but the sample size.
- It was also, briefly, being measured against a Imperium that held all
  fourteen fleet markers from turn 1 — markers the order of battle should have
  made it wait for (see the rules audit above). With the schedule enforced the
  effect survives, but a behaviour that spends a scarce resource has to be
  measured against the real scarcity.

The default is side-aware — the Imperium pickets, the Zhodani does not — and
forcing both sides to picket is **10.8 ± 2.6 VP worse** than that default, so
the asymmetry is doing real work. `tools/picket_check.py` reruns the whole
comparison, reporting the worlds claimed as well as the margin, because the
direct tally is far less noisy than a game result. Both halves can be switched
off independently: `pickets=False` stops the detachment, `free_world=0` stops
free worlds pulling on ordinary fleets.

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
