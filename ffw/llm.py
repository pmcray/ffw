"""Doctrine proposed in words by a language model, graded by the engine.

Every other trainer here searches numbers: cross-entropy over a weight vector,
gradient descent over a network, sampled rollouts over a shortlist.  None of
them can propose *a behaviour that does not yet exist*.  The Imperium left 55
victory points of undefended neutral worlds unclaimed for the whole project not
because a coefficient was wrong but because nothing in the search space said
"detach one squadron and go and sit on it"; a person had to notice.

This module closes that loop.  A language model reads a briefing -- the current
rule set, how it actually performed, where the victory points went, which rules
never fired -- and proposes edits in the ``DoctrineAgent`` rule vocabulary.  The
engine then does what no model's opinion can substitute for: plays the candidate
against the incumbent over paired games and reports an advantage with a standard
error.  Proposals are accepted only if they win, and the measurement -- win or
lose -- goes into the next briefing.

The division of labour is the whole point.  The model supplies imagination over
a space that numeric search cannot reach; the engine supplies the only thing a
model cannot supply about its own ideas, which is whether they are true.

    export ANTHROPIC_API_KEY=...
    python tools/evolve_doctrine.py --generations 5 --proposals 4 --games 12

``StubProposer`` drives the identical loop with scripted proposals, so the
machinery is testable without a key or a network.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .agents.doctrine import (DoctrineAgent, EFFECTS, ENUMS, FLAGS, NUMBERS,
                              POSTURE_KEYS, RuleError, RuleSet, default_rules)
from .engine import new_game, play
from .state import IMPERIAL, ZHODANI
from .training import AgentSpec, evaluate_paired

MODEL = "claude-opus-5"


# --------------------------------------------------------------------------
# The wire format.  Everything the model emits is a string or a list of
# strings: that stays inside the subset of JSON Schema structured outputs
# supports, and it keeps the proposals readable in the transcript.
# --------------------------------------------------------------------------
PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "hypothesis": {
                        "type": "string",
                        "description": "What should measurably change, and "
                                       "roughly by how much, if this is right.",
                    },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string",
                                           "enum": ["add", "remove",
                                                    "replace", "posture"]},
                                "name": {"type": "string"},
                                "when": {"type": "array",
                                         "items": {"type": "string"}},
                                "then": {"type": "string"},
                                "rationale": {"type": "string"},
                                "set": {"type": "array",
                                        "items": {"type": "string"}},
                            },
                            "required": ["action", "name", "when", "then",
                                         "rationale", "set"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["title", "hypothesis", "edits"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["proposals"],
    "additionalProperties": False,
}


def parse_effect(value):
    """``"prefer 2.5"`` / ``"avoid 1"`` / ``"veto"`` -> the stored dict form."""
    if isinstance(value, dict):
        return value
    text = str(value or "").strip().lower()
    if not text:
        raise RuleError("a rule needs an effect (%s)" % "/".join(EFFECTS))
    parts = text.split()
    effect = parts[0]
    if effect == "veto":
        return {"veto": True}
    if effect not in EFFECTS:
        raise RuleError("unknown effect %r (known: %s)"
                        % (effect, "/".join(EFFECTS)))
    if len(parts) < 2:
        raise RuleError("%s needs a strength, e.g. '%s 1.5'" % (effect, effect))
    try:
        return {effect: float(parts[1])}
    except ValueError:
        raise RuleError("%s expects a number, got %r" % (effect, parts[1])) from None


def parse_posture(items) -> dict:
    """``["pickets=false", "disengage_ratio=2.2"]`` -> a posture dict."""
    out = {}
    for item in items or []:
        if "=" not in str(item):
            raise RuleError("posture must read key=value, got %r" % (item,))
        key, _, value = str(item).partition("=")
        key, value = key.strip(), value.strip()
        if key not in POSTURE_KEYS:
            raise RuleError("unknown posture %r (known: %s)"
                            % (key, ", ".join(sorted(POSTURE_KEYS))))
        if POSTURE_KEYS[key] is bool:
            out[key] = value.lower() in ("1", "true", "yes", "on")
        else:
            try:
                out[key] = float(value)
            except ValueError:
                raise RuleError("posture %s expects a number, got %r"
                                % (key, value)) from None
    return out


def to_edits(raw) -> list:
    """Convert one proposal's wire edits into ``RuleSet.apply`` edits."""
    edits = []
    for item in raw:
        action = str(item.get("action", "")).strip()
        if action == "posture":
            edits.append({"action": "posture",
                          "posture": parse_posture(item.get("set"))})
        elif action == "remove":
            edits.append({"action": "remove", "name": item.get("name", "")})
        elif action in ("add", "replace"):
            edits.append({"action": action,
                          "rule": {"name": item.get("name", ""),
                                   "when": item.get("when") or [],
                                   "then": parse_effect(item.get("then")),
                                   "rationale": item.get("rationale", "")}})
        else:
            raise RuleError("unknown edit action %r" % (action,))
    return edits


@dataclass
class Proposal:
    title: str
    hypothesis: str
    edits: list
    ruleset: RuleSet | None = None
    error: str | None = None
    advantage: float | None = None
    stderr: float | None = None
    verdict: str = "unevaluated"
    accepted: bool = False
    confirmation: dict | None = None

    def summary(self) -> str:
        if self.error:
            return "%-44s REJECTED (%s)" % (self.title[:44], self.error)
        return ("%-44s %+7.1f +/- %5.1f  %s%s"
                % (self.title[:44], self.advantage or 0.0, self.stderr or 0.0,
                   self.verdict, "  ACCEPTED" if self.accepted else ""))


# --------------------------------------------------------------------------
def vocabulary() -> str:
    """The condition language, generated from the parser so it cannot drift."""
    lines = ["FLAGS (bare, or negated with a leading '!'):"]
    lines += ["  " + ", ".join(FLAGS)]
    lines.append("ENUMS (key=value):")
    for key, values in ENUMS.items():
        lines.append("  %s = %s" % (key, " | ".join(values)))
    lines.append("MEASURES (key>=n or key<=n; bare name means 'greater than zero'):")
    lines += ["  " + ", ".join(NUMBERS)]
    lines.append("EFFECTS (exactly one per rule): "
                 "'prefer <n>', 'avoid <n>', or 'veto'")
    lines.append("POSTURES (set as key=value): "
                 + ", ".join("%s (%s)" % (k, v.__name__)
                             for k, v in sorted(POSTURE_KEYS.items())))
    return "\n".join(lines)


SYSTEM_PROMPT = """You are advising on the doctrine of a fleet commander in \
Fifth Frontier War (GDW, 1981), the Zhodani invasion of the Imperial Spinward \
Marches. Doctrine is a list of named rules; each rule is a conjunction of \
conditions and one effect on how attractive a destination is. Your job is to \
propose edits to that rule set.

CONDITION VOCABULARY -- a condition outside this list is rejected unparsed, and \
a rejected proposal is a wasted generation:

%s

Rules are evaluated per candidate destination for one fleet. Every condition in \
a rule must hold for it to fire; the effects of all firing rules are summed, \
and any firing veto disqualifies the destination outright.

HOW TO PROPOSE WELL

Each proposal is one idea, expressed as the smallest set of edits that tests \
it. Do not bundle three unrelated changes into one proposal -- when the result \
comes back you will not know which part worked.

Prefer conjunctions over strengths. Raising a number is what the numeric \
trainers already do, and they do it better than you can by inspection. Your \
advantage is that you can say something they cannot express at all: a condition \
combination that has never been in the rule set. Reach for a new `when` clause \
before you reach for a bigger `prefer`.

State a hypothesis that could turn out false, in terms of a number in the \
briefing -- "the Imperium should end holding 3 more of the undefended neutrals, \
worth about 8 VP" -- not "this should improve play". You will be shown the \
measured result, and a hypothesis you cannot check against it teaches nobody \
anything.

Read the diagnostics before the rule set. The briefing reports where the \
victory points actually went and which rules never fired once in a whole war. A \
rule that never fires is not doctrine, it is a comment: either its conditions \
are unreachable, in which case fix or remove it, or the situation never arises, \
in which case it is not where the points are. Points nobody is collecting are \
worth more attention than points being contested.

Respect the game. Troops are the scarce resource, not warships. A world is only \
held if a garrison stands on it. A fleet that jumps somewhere it cannot refuel \
is out of the war permanently. The side you are advising may be structurally \
behind -- the Zhodani hold a six-to-one sealift advantage from the printed \
order of battle -- and no rule fixes that; find the points that are actually \
available.""" % vocabulary()


# --------------------------------------------------------------------------
def diagnose(ruleset: RuleSet, games: int = 4, max_turns: int = 40,
             seed: int = 200) -> dict:
    """Play the incumbent against itself and report where the points went.

    This is the substance of the briefing.  A model handed only a rule set can
    do nothing but rewrite it prettier; a model shown that eleven undefended
    worlds finish unclaimed has somewhere to aim.
    """
    tally = {IMPERIAL: 0.0, ZHODANI: 0.0, "unclaimed": 0.0}
    free = {IMPERIAL: 0.0, ZHODANI: 0.0, "unclaimed": 0.0}
    taken = {IMPERIAL: 0.0, ZHODANI: 0.0}
    margins = []
    coverage: dict[str, int] = {}
    for g in range(games):
        state = new_game(seed=seed + g)
        imperial = DoctrineAgent(IMPERIAL, ruleset, seed=g)
        zhodani = DoctrineAgent(ZHODANI, ruleset, seed=g + 1)
        play(state, imperial, zhodani, max_turns=max_turns)
        margins.append(state.victory_margin())
        for agent in (imperial, zhodani):
            for name, count in agent.coverage().items():
                coverage[name] = coverage.get(name, 0) + count
        for world in state.world_map:
            holder = world.control
            origin = world.original_control
            if world.owner == "neutral":
                bucket = free if (not world.defense_battalions and not world.sdb) \
                    else tally
                bucket[holder if holder else "unclaimed"] += 1.0
            elif holder is not None and holder != origin:
                taken[holder] += world.victory_points
    scale = float(max(1, games))
    mean = sum(margins) / len(margins)
    spread = (sum((m - mean) ** 2 for m in margins) / max(1, len(margins) - 1)) ** 0.5
    return {
        "games": games,
        "mean_margin": mean,
        "margin_spread": spread,
        "margins": margins,
        "free_worlds": {k: v / scale for k, v in free.items()},
        "other_neutrals": {k: v / scale for k, v in tally.items()},
        "vp_from_conquest": {k: v / scale for k, v in taken.items()},
        "silent_rules": [r.name for r in ruleset.rules
                         if coverage.get(r.name, 0) == 0],
        "busiest_rules": sorted(coverage.items(), key=lambda kv: -kv[1])[:5],
    }


def evaluate(ruleset: RuleSet, against, games: int = 40, seed: int = 7000,
             max_turns: int = 40, workers=None) -> dict:
    """Paired games, both seats, shared seeds: does this rule set play better?

    Forty games rather than a handful.  At twelve the standard error is about
    six victory points -- wider than most doctrine changes are worth -- and a
    loop that accepts on that is curating noise.
    """
    return evaluate_paired(spec_for(ruleset), against, games=games, seed=seed,
                           max_turns=max_turns, workers=workers)


def spec_for(ruleset: RuleSet) -> AgentSpec:
    """A picklable factory for a rule set, so candidates grade in parallel."""
    return AgentSpec(kind="doctrine", rules=ruleset.to_dict())


# --------------------------------------------------------------------------
def brief(ruleset: RuleSet, diagnostics: dict, benchmark: dict,
          history: list, wanted: int) -> str:
    """Everything the model gets to see, in the order it should read it."""
    free = diagnostics["free_worlds"]
    conquest = diagnostics["vp_from_conquest"]
    lines = [
        "## How the current doctrine actually performs",
        "",
        "Against the stock scripted agent, paired over both seats: "
        "%+.1f +/- %.1f VP (%s)."
        % (benchmark["advantage"], benchmark["stderr"], benchmark["verdict"]),
        "Self-play over %d games: mean margin %+.1f VP (spread %.1f); the "
        "margin is Zhodani-positive, and the Zhodani are structurally favoured."
        % (diagnostics["games"], diagnostics["mean_margin"],
           diagnostics["margin_spread"]),
        "",
        "## Where the victory points went, per game",
        "",
        "Victory points taken by conquest: Imperial %.1f, Zhodani %.1f."
        % (conquest[IMPERIAL], conquest[ZHODANI]),
        "Of the 18 undefended neutral worlds (55 VP, no defences at all): "
        "Imperial holds %.1f, Zhodani %.1f, and %.1f finish claimed by nobody."
        % (free.get(IMPERIAL, 0.0), free.get(ZHODANI, 0.0),
           free.get("unclaimed", 0.0)),
        "",
        "## Rule coverage",
        "",
        "Rules that never fired in any game: %s"
        % (", ".join(diagnostics["silent_rules"]) or "(none)"),
        "Busiest rules: %s"
        % ", ".join("%s (%d)" % (n, c) for n, c in diagnostics["busiest_rules"]),
        "",
        "## The current rule set",
        "",
        "```json",
        json.dumps(ruleset.to_dict(), indent=1),
        "```",
        "",
    ]
    if history:
        lines += ["## What has already been tried, and what it measured", ""]
        for entry in history[-14:]:
            lines.append("- %s" % entry)
        lines.append("")
        lines.append("Do not re-propose an idea that has already been measured "
                     "unless you are changing something specific about why it "
                     "failed, and say what.")
        lines.append("")
    lines += [
        "## Your task",
        "",
        "Propose %d independent candidate edits to this doctrine. Each is "
        "evaluated on its own, against the current rule set, over paired games "
        "on shared seeds -- so make them independent ideas, not a sequence that "
        "only works applied together." % wanted,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
class StubProposer:
    """A scripted proposer, so the loop can be tested without a network.

    It exercises the same path as the real one: the same wire format, the same
    parsing, the same validation, the same rejection on a bad condition.
    """

    def __init__(self, script):
        self.script = list(script)
        self.briefings: list[str] = []

    def propose(self, briefing: str, wanted: int) -> list:
        self.briefings.append(briefing)
        batch = self.script.pop(0) if self.script else []
        return [Proposal(p.get("title", "untitled"), p.get("hypothesis", ""),
                         p.get("edits", [])) for p in batch]


class ClaudeProposer:
    """Asks Claude for doctrine edits, in the rule vocabulary."""

    def __init__(self, model: str = MODEL, effort: str = "high",
                 client=None, max_tokens: int = 16000):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.transcript: list[dict] = []
        if client is not None:
            self.client = client
        else:
            try:
                import anthropic
            except ImportError:  # pragma: no cover - depends on the environment
                raise RuntimeError(
                    "the proposal loop needs the Anthropic SDK: "
                    "pip install anthropic") from None
            try:
                self.client = anthropic.Anthropic()
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "could not authenticate: set ANTHROPIC_API_KEY, or run "
                    "`ant auth login`.  (%s)" % exc) from None

    def propose(self, briefing: str, wanted: int) -> list:
        response = self._call(briefing)
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise RuntimeError("the request was declined (%s)"
                               % getattr(details, "category", "unknown"))
        text = next((b.text for b in response.content
                     if getattr(b, "type", None) == "text"), "")
        self.transcript.append({"briefing": briefing, "reply": text})
        payload = json.loads(text)
        return [Proposal(p.get("title", "untitled"), p.get("hypothesis", ""),
                         p.get("edits", []))
                for p in payload.get("proposals", [])][:wanted]

    def _call(self, briefing: str):
        system = [{"type": "text", "text": SYSTEM_PROMPT,
                   "cache_control": {"type": "ephemeral"}}]
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "output_config": {"effort": self.effort,
                              "format": {"type": "json_schema",
                                         "schema": PROPOSAL_SCHEMA}},
            "messages": [{"role": "user", "content": briefing}],
        }
        # Claude Opus 5's safety classifiers can decline a request outright, so
        # ask the API to re-serve a decline on its recommended fallback model
        # rather than losing the generation.  Older SDKs do not know the
        # parameter; fall back to the plain endpoint rather than failing.
        try:
            return self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default", **request)
        except TypeError:
            return self.client.messages.create(**request)


# --------------------------------------------------------------------------
@dataclass
class EvolutionLog:
    rows: list = field(default_factory=list)

    def add(self, **kw) -> None:
        self.rows.append(kw)

    def history(self) -> list:
        out = []
        for row in self.rows:
            if row.get("error"):
                out.append("%s -- rejected before it was played: %s"
                           % (row["title"], row["error"]))
            else:
                line = ("%s -- %+.1f +/- %.1f VP over %d paired games (%s)"
                        % (row["title"], row["advantage"], row["stderr"],
                           row["games"], row["verdict"]))
                check = row.get("confirmation")
                if check is not None:
                    line += ("; re-measured on fresh seeds at %+.1f +/- %.1f"
                             % (check["advantage"], check["stderr"]))
                if row["accepted"]:
                    line += "; accepted"
                out.append(line)
        return out


def evolve(generations: int = 4, proposals: int = 4, games: int = 40,
           ruleset: RuleSet | None = None, proposer=None, opponent=None,
           max_turns: int = 40, seed: int = 7000, diagnostic_games: int = 4,
           accept_sigma: float = 1.0, confirm_games: int = 0, progress=None):
    """Propose, measure, keep what survives.  Returns ``(ruleset, log)``.

    Two guards, both learned the expensive way on this game.

    ``accept_sigma`` is the bar a winner has to clear, in standard errors.  At
    zero the loop accepts whichever candidate the seeds happened to favour and
    walks a random path dressed up as progress; at two almost nothing clears the
    bar at any affordable sample size and the loop never moves.  One is the
    compromise, and the verdict is reported alongside so a weak acceptance is
    visible as one.

    ``confirm_games`` re-measures the winner on **fresh seeds** before committing
    it.  A candidate chosen as the best of four on one set of games has been
    selected partly for fitting those games -- exactly the trap that made a
    ten-game evaluator comparison read +10.2 and a sixty-game one read +1.6.
    Re-rolling the dice is cheap next to believing a winner that isn't one.
    """
    incumbent = (ruleset or default_rules()).copy()
    proposer = proposer or ClaudeProposer()
    against = opponent or AgentSpec(kind="scripted")
    log = EvolutionLog()

    for generation in range(generations):
        diagnostics = diagnose(incumbent, games=diagnostic_games,
                               max_turns=max_turns, seed=200 + generation * 17)
        benchmark = evaluate(incumbent, against, games=games,
                             seed=seed + generation * 101, max_turns=max_turns)
        briefing = brief(incumbent, diagnostics, benchmark, log.history(),
                         proposals)
        if progress is not None:
            progress("generation", generation, benchmark, None)

        batch = proposer.propose(briefing, proposals)
        best, best_gain = None, 0.0
        for proposal in batch:
            try:
                proposal.ruleset = incumbent.apply(to_edits(proposal.edits))
            except (RuleError, KeyError, TypeError, ValueError) as exc:
                proposal.error = str(exc)
                log.add(generation=generation, title=proposal.title,
                        error=proposal.error, accepted=False)
                if progress is not None:
                    progress("proposal", generation, None, proposal)
                continue
            # Common random numbers: every candidate in a generation is judged
            # on the same games, so the comparison is between doctrines rather
            # than between the seeds they happened to draw.
            result = evaluate(proposal.ruleset, spec_for(incumbent),
                              games=games, seed=seed + generation * 101,
                              max_turns=max_turns)
            proposal.advantage = result["advantage"]
            proposal.stderr = result["stderr"]
            proposal.verdict = result["verdict"]
            clears = result["advantage"] > accept_sigma * result["stderr"]
            if clears and result["advantage"] > best_gain:
                best, best_gain = proposal, result["advantage"]
            log.add(generation=generation, title=proposal.title,
                    hypothesis=proposal.hypothesis, error=None,
                    advantage=result["advantage"], stderr=result["stderr"],
                    games=result["games"], verdict=result["verdict"],
                    accepted=False, edits=proposal.edits)
            if progress is not None:
                progress("proposal", generation, None, proposal)

        if best is not None and confirm_games:
            # Fresh seeds: the winner was picked partly for fitting the games it
            # was picked on, so make it win again on games it has never seen.
            check = evaluate(best.ruleset, spec_for(incumbent),
                             games=confirm_games,
                             seed=seed + 500_000 + generation * 101,
                             max_turns=max_turns)
            best.confirmation = check
            if check["advantage"] <= 0:
                best.verdict = ("%s, but did not survive confirmation "
                                "(%+.1f on fresh seeds)"
                                % (best.verdict, check["advantage"]))
                for row in log.rows:
                    if (row.get("title") == best.title
                            and row["generation"] == generation):
                        row["confirmation"] = check
                        row["verdict"] = best.verdict
                if progress is not None:
                    progress("rejected", generation, None, best)
                best = None

        if best is not None:
            best.accepted = True
            incumbent = best.ruleset
            incumbent.notes = ("evolved: %s (%+.1f VP)"
                               % (best.title, best.advantage))
            for row in log.rows:
                if row.get("title") == best.title and row["generation"] == generation:
                    row["accepted"] = True
                    if best.confirmation is not None:
                        row["confirmation"] = best.confirmation
            if progress is not None:
                progress("accepted", generation, None, best)

    return incumbent, log


def save_run(path: str, ruleset: RuleSet, log: EvolutionLog,
             meta: dict | None = None) -> str:
    payload = {"ruleset": ruleset.to_dict(),
               "log": [{k: v for k, v in row.items()} for row in log.rows],
               "meta": dict(meta or {}, saved_at=time.strftime("%Y-%m-%d"))}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return path
