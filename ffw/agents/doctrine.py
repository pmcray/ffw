"""A rule-based agent whose doctrine is written in words rather than weights.

The other four agents are one idea wearing four hats: every decision is a
weighted sum of the same eighteen-odd features, and training moves the
coefficients.  That search space cannot express a conjunction.  "An undefended
neutral within two jumps that I am carrying troops for" is a single, obvious
instruction to a human commander, and there is no vector of per-feature weights
that says it -- a weight on ``undefended`` applies to defended worlds' absence
of defences everywhere, at every distance, carrying or empty.

``DoctrineAgent`` scores a destination by matching it against an ordered list of
**rules**, each a conjunction of named conditions and an effect:

    {"name": "occupy free worlds nearby",
     "when": ["claimable", "carrying>=1", "distance<=2"],
     "then": {"prefer": 2.6}}

The rule set is data, not code: it loads from JSON, reads like doctrine, and can
be written by hand or proposed by a language model and then graded empirically
(``ffw.llm``).  That is the point of the architecture -- the search space is
*behaviours*, and a proposal can be read and argued with before it is measured.

What is deliberately **not** inherited: the scoring.  ``score_destination`` is
overridden outright, so no term of the linear doctrine survives -- an empty rule
set scores every destination identically, which the tests assert.  What *is*
inherited from ``HeuristicAgent`` is movement plumbing that is not doctrine at
all: which hexes are in jump range, whether a fleet will find fuel when it
arrives, how a picket closes a multi-jump gap.  Those are rules of the game.
"""

from __future__ import annotations

import json
import math
import re

from .. import hexmap
from ..state import GameState, IMPERIAL, ZHODANI
from .heuristic import HeuristicAgent

#: conditions that stand alone (optionally negated with a leading "!")
FLAGS = (
    "undefended",       # no system defence boats and no defence battalions left
    "vacuum",           # airless: rule 5 lets a warship hold it with no troops
    "capital",          # subsector capital, worth double
    "base",             # naval, scout or military base
    "gas_giant",        # refuels anyone, friend or foe
    "water",            # refuels streamlined hulls
    "enemy_present",    # enemy squadrons in the hex
    "enemy_troops",     # enemy ground units on the surface
    "garrison_short",   # my world, garrison below what holding it needs
    "claimable",        # changes hands for the price of showing up
    "carrying",         # the fleet has troops with it (shorthand for carrying>=1)
    "red_zone",         # interdicted; the Imperial navy needs a warrant
)

#: conditions of the form ``key=value``
ENUMS = {
    "control": ("mine", "enemy", "unclaimed"),
    "owner": ("mine", "enemy", "neutral"),
}

#: conditions of the form ``key>=n`` or ``key<=n``
NUMBERS = (
    "tech",             # world tech level
    "vp",               # victory points the world is worth
    "distance",         # parsecs from where the fleet is starting
    "front",            # parsecs to the nearest enemy home world
    "turn",             # game turn
    "margin",           # victory margin in *my* favour
    "carrying",         # troop factors travelling with the fleet
    "support",          # my attack strength within one parsec
    "enemy_strength",   # enemy attack strength in the hex
    "garrison_need",    # troop factors needed to hold the world
)

#: effects a rule may have
EFFECTS = ("prefer", "avoid", "veto")

#: discrete behaviours a rule set can switch, outside the scoring
POSTURE_KEYS = {
    "pickets": bool,             # detach single squadrons onto free worlds
    "disengage_ratio": float,    # break off when outgunned by this much
    "guerrilla_boldness": float, # how readily Zhodani guerrillas rise
    "armistice_margin": float,   # Zhodani: call it off when ahead by this
    "bomb_defenses_first": bool, # soften battalions before enemy troops
}

DEFAULT_POSTURE = {
    "pickets": True,
    "disengage_ratio": 1.9,
    "guerrilla_boldness": 0.35,
    "armistice_margin": 60.0,
    "bomb_defenses_first": True,
}

_CONDITION = re.compile(r"^(!?)([a-z_]+)(?:(=|>=|<=)(-?[\w.]+))?$")


class RuleError(ValueError):
    """A rule that does not parse, or names something outside the vocabulary."""


# --------------------------------------------------------------------------
class Condition:
    """One clause of a rule, parsed from its written form."""

    __slots__ = ("text", "key", "op", "value", "negated")

    def __init__(self, text: str):
        match = _CONDITION.match(text.strip())
        if not match:
            raise RuleError("cannot parse condition %r" % (text,))
        negated, key, op, value = match.groups()
        self.text = text.strip()
        self.key = key
        self.negated = negated == "!"
        self.op = op
        if op is None:
            if key not in FLAGS and key not in NUMBERS:
                raise RuleError("unknown flag %r (known: %s)"
                                % (key, ", ".join(FLAGS)))
            self.value = None
        elif op == "=":
            if key not in ENUMS:
                raise RuleError("unknown enum %r (known: %s)"
                                % (key, ", ".join(ENUMS)))
            if value not in ENUMS[key]:
                raise RuleError("%s must be one of %s, not %r"
                                % (key, "/".join(ENUMS[key]), value))
            self.value = value
        else:
            if key not in NUMBERS:
                raise RuleError("unknown measure %r (known: %s)"
                                % (key, ", ".join(NUMBERS)))
            try:
                self.value = float(value)
            except (TypeError, ValueError):
                raise RuleError("%s%s expects a number, got %r"
                                % (key, op, value)) from None

    def holds(self, facts: dict) -> bool:
        if self.op is None:
            # a bare measure means "greater than zero", so `carrying` reads the
            # way a person would say it
            truth = bool(facts.get(self.key))
            return truth != self.negated
        actual = facts.get(self.key)
        if actual is None:
            return self.negated
        if self.op == "=":
            truth = actual == self.value
        elif self.op == ">=":
            truth = float(actual) >= self.value
        else:
            truth = float(actual) <= self.value
        return truth != self.negated

    def __repr__(self) -> str:
        return "<%s>" % self.text


class Rule:
    """A named conjunction of conditions and what it does to a destination."""

    __slots__ = ("name", "rationale", "when", "effect", "amount")

    def __init__(self, name: str, when, then: dict, rationale: str = ""):
        if not name or not str(name).strip():
            raise RuleError("every rule needs a name")
        self.name = str(name).strip()
        self.rationale = str(rationale or "")
        if isinstance(when, str):
            when = [when]
        if not when:
            raise RuleError("rule %r has no conditions; a rule that matches "
                            "everything is a weight, not a rule" % (self.name,))
        self.when = [Condition(c) for c in when]
        if not isinstance(then, dict) or len(then) != 1:
            raise RuleError("rule %r needs exactly one effect from %s"
                            % (self.name, "/".join(EFFECTS)))
        (effect, amount), = then.items()
        if effect not in EFFECTS:
            raise RuleError("unknown effect %r in rule %r (known: %s)"
                            % (effect, self.name, "/".join(EFFECTS)))
        self.effect = effect
        if effect == "veto":
            self.amount = 0.0
        else:
            try:
                self.amount = float(amount)
            except (TypeError, ValueError):
                raise RuleError("rule %r: %s expects a number, got %r"
                                % (self.name, effect, amount)) from None

    def matches(self, facts: dict) -> bool:
        return all(c.holds(facts) for c in self.when)

    def to_dict(self) -> dict:
        out = {"name": self.name,
               "when": [c.text for c in self.when],
               "then": ({"veto": True} if self.effect == "veto"
                        else {self.effect: self.amount})}
        if self.rationale:
            out["rationale"] = self.rationale
        return out

    def __repr__(self) -> str:
        return "<Rule %r %s>" % (self.name, [c.text for c in self.when])


class RuleSet:
    """An ordered list of rules plus the postures they do not cover."""

    def __init__(self, rules=None, posture=None, notes: str = ""):
        self.rules = list(rules or [])
        self.posture = dict(DEFAULT_POSTURE)
        for key, value in (posture or {}).items():
            if key not in POSTURE_KEYS:
                raise RuleError("unknown posture %r (known: %s)"
                                % (key, ", ".join(sorted(POSTURE_KEYS))))
            self.posture[key] = POSTURE_KEYS[key](value)
        self.notes = notes

    # -- construction ---------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> "RuleSet":
        rules = [Rule(r.get("name", ""), r.get("when", []), r.get("then", {}),
                      r.get("rationale", "")) for r in data.get("rules", [])]
        names = [r.name for r in rules]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise RuleError("duplicate rule names: %s"
                            % ", ".join(sorted(duplicates)))
        return cls(rules, data.get("posture"), data.get("notes", ""))

    def to_dict(self) -> dict:
        return {"notes": self.notes,
                "posture": dict(self.posture),
                "rules": [r.to_dict() for r in self.rules]}

    @classmethod
    def load(cls, path: str) -> "RuleSet":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)
        return path

    def copy(self) -> "RuleSet":
        return RuleSet.from_dict(self.to_dict())

    # -- editing (what a proposal does) ---------------------------------
    def apply(self, edits) -> "RuleSet":
        """Return a copy with a list of add / remove / replace edits applied.

        Edits are the unit a proposal deals in, so a language model changes one
        named behaviour at a time and the change can be attributed when the
        result comes back.
        """
        new = self.copy()
        by_name = {r.name: i for i, r in enumerate(new.rules)}
        for edit in edits:
            action = edit.get("action")
            if action == "remove":
                name = edit.get("name", "")
                if name not in by_name:
                    raise RuleError("cannot remove unknown rule %r" % (name,))
                del new.rules[by_name[name]]
                by_name = {r.name: i for i, r in enumerate(new.rules)}
            elif action in ("add", "replace"):
                spec = edit.get("rule") or {}
                rule = Rule(spec.get("name", ""), spec.get("when", []),
                            spec.get("then", {}), spec.get("rationale", ""))
                if rule.name in by_name:
                    if action == "add":
                        raise RuleError("rule %r already exists; use replace"
                                        % (rule.name,))
                    new.rules[by_name[rule.name]] = rule
                else:
                    if action == "replace":
                        raise RuleError("cannot replace unknown rule %r"
                                        % (rule.name,))
                    new.rules.append(rule)
                    by_name[rule.name] = len(new.rules) - 1
            elif action == "posture":
                for key, value in (edit.get("posture") or {}).items():
                    if key not in POSTURE_KEYS:
                        raise RuleError("unknown posture %r" % (key,))
                    new.posture[key] = POSTURE_KEYS[key](value)
            else:
                raise RuleError("unknown edit action %r" % (action,))
        return new

    # -- scoring --------------------------------------------------------
    def score(self, facts: dict):
        """``(score, matched_rule_names)`` -- or ``None`` if a rule vetoed it."""
        total = 0.0
        fired = []
        for rule in self.rules:
            if not rule.matches(facts):
                continue
            fired.append(rule.name)
            if rule.effect == "veto":
                return None, fired
            total += rule.amount if rule.effect == "prefer" else -rule.amount
        return total, fired

    def __len__(self) -> int:
        return len(self.rules)

    def __repr__(self) -> str:
        return "<RuleSet %d rules>" % len(self.rules)


# --------------------------------------------------------------------------
#: A doctrine written out in rules.  This is the hand-written starting point --
#: roughly what the linear doctrine knows, said in conjunctions -- and it is
#: what any proposed rule set has to beat.
DEFAULT_RULES = {
    "notes": "Hand-written seed doctrine: the baseline a proposal must beat.",
    "posture": dict(DEFAULT_POSTURE),
    "rules": [
        {"name": "take enemy worlds worth taking",
         "when": ["control=enemy", "carrying>=1"],
         "then": {"prefer": 1.4},
         "rationale": "Enemy worlds are where the victory points are."},
        {"name": "weight the prize by its value",
         "when": ["control=enemy", "vp>=10"],
         "then": {"prefer": 1.1},
         "rationale": "A tech-10 world is worth ten times a tech-1 one."},
        {"name": "subsector capitals are worth double",
         "when": ["capital", "!control=mine"],
         "then": {"prefer": 1.2},
         "rationale": "Rule 8 pays twice the tech level for a capital."},
        {"name": "do not storm what the landing force cannot hold",
         "when": ["control=enemy", "carrying<=20", "garrison_need>=25"],
         "then": {"avoid": 2.0},
         "rationale": "A landing that cannot beat the garrison wastes the army."},
        {"name": "occupy free worlds within reach",
         "when": ["claimable", "carrying>=1", "distance<=3"],
         "then": {"prefer": 2.4},
         "rationale": "Eighteen neutrals have no defences at all and carry 55 VP."},
        {"name": "airless free worlds need no troops at all",
         "when": ["claimable", "vacuum", "distance<=3"],
         "then": {"prefer": 2.0},
         "rationale": "Rule 5: a warship may garrison a surrendered vacuum world."},
        {"name": "avoid a stand-up fight with a stronger fleet",
         "when": ["enemy_present", "enemy_strength>=30"],
         "then": {"avoid": 1.6},
         "rationale": "Losing the fleet loses the war; the worlds keep."},
        {"name": "press an enemy fleet when the odds are mine",
         "when": ["enemy_present", "enemy_strength<=15", "support>=25"],
         "then": {"prefer": 1.2},
         "rationale": "Concentration is the one advantage worth spending."},
        {"name": "never jump a loaded invasion into a stronger fleet",
         "when": ["carrying>=100", "enemy_present", "enemy_strength>=25"],
         "then": {"veto": True},
         "rationale": "Transports die with their cargo; there is no recovering it."},
        {"name": "cover my own worlds when their garrison is short",
         "when": ["control=mine", "garrison_short"],
         "then": {"prefer": 1.0},
         "rationale": "A world whose garrison lapses reverts to its owner."},
        {"name": "advance towards the enemy's own space",
         "when": ["front<=4"],
         "then": {"prefer": 0.9},
         "rationale": "Without a gradient the front freezes and fleets shuffle."},
        {"name": "prefer somewhere I can refuel",
         "when": ["gas_giant"],
         "then": {"prefer": 0.8},
         "rationale": "A fleet that cannot refuel is out of the war."},
        {"name": "friendly bases are worth calling at",
         "when": ["control=mine", "base"],
         "then": {"prefer": 0.5}},
        {"name": "long jumps cost time",
         "when": ["distance>=4"],
         "then": {"avoid": 0.7},
         "rationale": "Time is the scarce resource in a 45-turn war."},
        {"name": "collect troops before an invasion",
         "when": ["control=mine", "!carrying", "!garrison_short"],
         "then": {"prefer": 0.4},
         "rationale": "An empty fleet should call where troops are waiting."},
        {"name": "the Imperial navy may not land in a red zone",
         "when": ["red_zone"],
         "then": {"avoid": 3.0},
         "rationale": "Landing needs a warrant; without one the trip is wasted."},
        {"name": "consolidate when well ahead",
         "when": ["margin>=120", "control=mine"],
         "then": {"prefer": 0.7},
         "rationale": "A won war is lost by overreaching in the last ten turns."},
        {"name": "press harder when behind",
         "when": ["margin<=-60", "control=enemy"],
         "then": {"prefer": 0.9},
         "rationale": "A stalemate is a loss for whoever needed the points."},
    ],
}


def default_rules() -> RuleSet:
    return RuleSet.from_dict(DEFAULT_RULES)


# --------------------------------------------------------------------------
class DoctrineAgent(HeuristicAgent):
    """Plays by a written rule set instead of a weight vector."""

    name = "doctrine"

    def __init__(self, side: str, rules=None, seed: int | None = None,
                 label: str | None = None, explain: bool = False):
        super().__init__(side, None, seed, label)
        if rules is None:
            self.ruleset = default_rules()
        elif isinstance(rules, RuleSet):
            self.ruleset = rules
        elif isinstance(rules, str):
            self.ruleset = RuleSet.load(rules)
        else:
            self.ruleset = RuleSet.from_dict(rules)
        self._pickets = bool(self.ruleset.posture["pickets"])
        #: rule -> how often it decided something, for the proposal briefing
        self.fired: dict[str, int] = {}
        self.explain = explain

    # -- the facts a rule can ask about ---------------------------------
    def facts(self, state: GameState, side: str, origin: str, dest: str,
              carrying: int, index: dict) -> dict:
        """Everything the rule vocabulary can name, for one candidate hex."""
        world = state.world_map.get(dest)
        enemy = state.enemy_of(side)
        control = ("mine" if world.control == side else
                   "enemy" if world.control == enemy else "unclaimed")
        owner = ("mine" if world.original_control == side else
                 "enemy" if world.original_control == enemy else "neutral")
        near = state.world_map.adjacent(dest)
        margin = state.victory_margin()
        garrison_need = world.garrison_required(side)
        return {
            "undefended": world.sdb_current == 0 and world.defense_current == 0,
            "vacuum": world.atmosphere == "vacuum",
            "capital": world.subsector_capital,
            "base": world.base,
            "gas_giant": world.gas_giant,
            "water": world.water,
            "enemy_present": bool(index["foe_count"].get(dest)),
            "enemy_troops": bool(state.troops_at(dest, enemy)),
            "garrison_short": (world.control == side and garrison_need >
                               state.garrison_strength(dest, side)),
            "claimable": self.claimable(state, side, world, carrying, index),
            "red_zone": (world.zone == "red" and side == IMPERIAL
                         and not world.zone_lifted),
            "control": control,
            "owner": owner,
            "tech": world.tech_level,
            "vp": world.victory_points,
            "distance": hexmap.distance(origin, dest),
            "front": self._distance_to_front(state, side, dest),
            "turn": state.turn,
            "margin": margin if side == ZHODANI else -margin,
            "carrying": carrying,
            "support": sum(index["own"].get(h, 0) for h in near),
            "enemy_strength": index["foe"].get(dest, 0),
            "garrison_need": garrison_need,
        }

    # -- scoring, entirely by rule --------------------------------------
    def score_destination(self, state, side, origin, dest, carrying=0,
                          index=None):
        world = state.world_map.get(dest)
        if world is None:
            return self._score_box(state, side, origin, dest, carrying)
        if index is None:
            index = self.squadron_index(state, side)
        facts = self.facts(state, side, origin, dest, carrying, index)
        score, fired = self.ruleset.score(facts)
        for name in fired:
            self.fired[name] = self.fired.get(name, 0) + 1
        if score is None:
            return -1e9
        return score

    def _score_box(self, state, side, origin, dest, carrying):
        """Reinforcement boxes are collection errands, not doctrine."""
        from ..engine import REINFORCEMENT_BOXES
        if REINFORCEMENT_BOXES.get(dest) != side:
            return -1e9
        waiting = sum(1 for sq in state.squadrons.values()
                      if sq.location == dest and sq.fleet is None
                      and sq.cls.kind != "scout")
        if waiting == 0 or carrying > 0:
            return -1e9
        return min(3.0, waiting / 3.0)

    def explain_choice(self, state, side, origin, dest, carrying=0) -> list:
        """Which rules fired on this destination, and by how much."""
        index = self.squadron_index(state, side)
        facts = self.facts(state, side, origin, dest, carrying, index)
        out = []
        for rule in self.ruleset.rules:
            if rule.matches(facts):
                out.append((rule.name, rule.effect,
                            rule.amount if rule.effect == "prefer" else -rule.amount))
        return out

    # -- the postures the rules do not cover ----------------------------
    def wants_to_disengage(self, state, side, hex_id, engine):
        own = state.squadrons_at(hex_id, side)
        foe = state.squadrons_at(hex_id, state.enemy_of(side))
        if not own or not foe:
            return False
        mine = sum(sq.attack for sq in own) + 1
        theirs = sum(sq.attack for sq in foe) + 1
        if any(sq.troops for sq in own) and theirs > mine:
            return True
        return theirs > mine * float(self.ruleset.posture["disengage_ratio"])

    def set_guerrilla_modes(self, state, engine):
        boldness = float(self.ruleset.posture["guerrilla_boldness"])
        for unit in state.troops.values():
            if not unit.guerrilla:
                continue
            world = state.world_map.get(unit.location)
            if world is None:
                continue
            opposition = (state.garrison_strength(unit.location, IMPERIAL)
                          + world.defense_current)
            unit.overt = (unit.current * (1.0 + boldness) > opposition * 0.6
                          and unit.losses < 60)

    def declare_armistice(self, state, engine):
        if self.side != ZHODANI:
            return False
        return (state.turn >= 40
                and state.victory_margin()
                > float(self.ruleset.posture["armistice_margin"]))

    def bombing_targets(self, state, side, hex_id, bombard, engine):
        if self.ruleset.posture["bomb_defenses_first"]:
            return super().bombing_targets(state, side, hex_id, bombard, engine)
        world = state.world_map.get(hex_id)
        enemy = state.enemy_of(side)
        troops = [t for t in state.troops_at(hex_id, enemy)
                  if not (t.guerrilla and not t.overt)]
        if not troops:
            return super().bombing_targets(state, side, hex_id, bombard, engine)
        troops.sort(key=lambda t: -t.current)
        take = troops[:2]
        return [(t, bombard // len(take)) for t in take]

    # -- reporting ------------------------------------------------------
    def coverage(self) -> dict:
        """How often each rule decided anything -- a rule that never fires is
        not doctrine, it is a comment, and the briefing says so."""
        return {r.name: self.fired.get(r.name, 0) for r in self.ruleset.rules}
