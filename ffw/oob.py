"""Orders of battle and counter definitions.

The force levels here come from the Imperial and Zhodani Order of Battle charts
(pages 23-24 of the scanned rules), which are printed as plain text and can be
followed exactly.  Individual counter values come from the three counter sheets
(pages 31-34).  The scans do not permit a reliable counter-by-counter reading of
all 720 pieces, so each category is given the *mix of counter classes actually
observed on the sheet*, expanded to the exact number of counters the order of
battle calls for.  Aggregate force levels therefore match the printed game; the
identity of any individual squadron may differ from a specific cardboard chit.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# squadron classes
# --------------------------------------------------------------------------
STREAMLINED, PARTIAL, UNSTREAMLINED = "streamlined", "partial", "unstreamlined"

#: troop capacity by squadron type, in combat factors (chart sheet)
TROOP_CAPACITY = {
    "assault": (600, 300),
    "battle": (20, 10),
    "cruiser": None,        # equal to the squadron's defence factor
    "scout": (0, 0),
    "tanker": (0, 0),
}


@dataclass(frozen=True)
class SquadronClass:
    name: str
    kind: str                      # battle / cruiser / assault / scout / tanker
    jump: int
    refuel: str
    attack: int
    bombard: int
    defense: int
    reduced: tuple[int, int, int] | None   # None = destroyed by any damage

    @property
    def code(self) -> str:
        return "%s%d" % (self.kind[0].upper(), self.jump)

    def capacity(self, reduced: bool) -> int:
        cap = TROOP_CAPACITY[self.kind]
        if cap is None:
            return self.reduced[2] if (reduced and self.reduced) else self.defense
        return cap[1] if reduced else cap[0]


def _sq(kind, jump, refuel, a, b, d, red):
    return SquadronClass("%s-%d-%d-%d-%d" % (kind, jump, a, b, d),
                         kind, jump, refuel, a, b, d, red)


# ---- Imperial ------------------------------------------------------------
IMPERIAL_BATTLE = [
    (_sq("battle", 4, UNSTREAMLINED, 6, 2, 8, (3, 1, 4)), 4),   # black globes
    (_sq("battle", 4, UNSTREAMLINED, 5, 1, 7, (2, 0, 3)), 6),
    (_sq("battle", 4, PARTIAL,       3, 1, 7, (1, 1, 4)), 3),
    (_sq("battle", 3, UNSTREAMLINED, 8, 0, 4, (4, 0, 2)), 6),
    (_sq("battle", 3, UNSTREAMLINED, 4, 4, 4, (2, 2, 2)), 6),
    (_sq("battle", 3, UNSTREAMLINED, 3, 3, 6, (1, 1, 3)), 2),
    (_sq("battle", 4, PARTIAL,       5, 4, 8, (2, 2, 4)), 5),
    (_sq("battle", 2, PARTIAL,       5, 1, 3, (2, 0, 1)), 2),
]
IMPERIAL_CRUISER = [
    (_sq("cruiser", 6, STREAMLINED, 4, 2, 8, (2, 1, 4)), 3),
    (_sq("cruiser", 5, PARTIAL,     5, 4, 6, (2, 2, 3)), 2),
    (_sq("cruiser", 5, PARTIAL,     3, 2, 4, (1, 1, 2)), 2),
    (_sq("cruiser", 4, STREAMLINED, 3, 4, 7, (1, 2, 3)), 6),
    (_sq("cruiser", 4, PARTIAL,     4, 4, 6, (2, 2, 3)), 5),
    (_sq("cruiser", 4, PARTIAL,     2, 2, 8, (1, 1, 4)), 3),
    (_sq("cruiser", 4, PARTIAL,     1, 2, 6, (0, 1, 3)), 3),
    (_sq("cruiser", 3, STREAMLINED, 3, 4, 6, (1, 2, 3)), 6),
    (_sq("cruiser", 2, PARTIAL,     2, 2, 6, (1, 1, 3)), 2),
]
IMPERIAL_SCOUT = [
    (_sq("scout", 4, STREAMLINED, 0, 0, 6, None), 4),
    (_sq("scout", 4, STREAMLINED, 0, 5, 6, None), 1),
    (_sq("scout", 3, STREAMLINED, 0, 8, 4, None), 1),
    (_sq("scout", 2, STREAMLINED, 0, 3, 4, None), 4),
    (_sq("scout", 2, STREAMLINED, 0, 8, 8, None), 1),
    (_sq("scout", 2, STREAMLINED, 0, 5, 6, None), 1),
]
IMPERIAL_TANKER = [(_sq("tanker", 3, PARTIAL, 0, 0, 4, (0, 0, 2)), 2)]
IMPERIAL_ASSAULT = [(_sq("assault", 4, STREAMLINED, 0, 0, 6, (0, 0, 3)), 7)]
IMPERIAL_COLONIAL_SQUADRONS = [
    (_sq("cruiser", 4, PARTIAL,     4, 4, 6, (2, 2, 3)), 6),
    (_sq("cruiser", 3, STREAMLINED, 3, 4, 6, (1, 2, 3)), 4),
    (_sq("battle", 3, UNSTREAMLINED, 3, 3, 6, (1, 1, 3)), 2),
    (_sq("cruiser", 2, PARTIAL,     2, 2, 6, (1, 1, 3)), 2),
]

# ---- Zhodani -------------------------------------------------------------
ZHODANI_BATTLE = [
    (_sq("battle", 3, UNSTREAMLINED, 6, 4, 8, (3, 2, 4)), 12),
    (_sq("battle", 3, UNSTREAMLINED, 3, 4, 8, (2, 2, 4)), 4),
    (_sq("battle", 3, UNSTREAMLINED, 3, 2, 7, (1, 1, 3)), 4),
    (_sq("battle", 3, UNSTREAMLINED, 3, 5, 9, (2, 2, 4)), 4),
    (_sq("battle", 4, UNSTREAMLINED, 4, 4, 9, (2, 2, 4)), 8),
    (_sq("battle", 3, UNSTREAMLINED, 2, 4, 8, (1, 2, 4)), 4),
    (_sq("battle", 3, UNSTREAMLINED, 3, 3, 7, (1, 1, 3)), 4),
    (_sq("battle", 2, UNSTREAMLINED, 3, 4, 8, (1, 2, 4)), 4),
    (_sq("battle", 2, UNSTREAMLINED, 1, 2, 4, (0, 1, 2)), 2),
]
ZHODANI_CRUISER = [
    (_sq("cruiser", 3, STREAMLINED, 5, 2, 5, (3, 1, 3)), 8),
    (_sq("cruiser", 3, STREAMLINED, 3, 1, 5, (1, 0, 3)), 4),
    (_sq("cruiser", 4, PARTIAL,     2, 5, 7, (1, 3, 4)), 6),
    (_sq("cruiser", 4, PARTIAL,     1, 1, 3, (0, 0, 2)), 3),
    (_sq("cruiser", 3, PARTIAL,     3, 3, 6, (1, 3, 4)), 4),
    (_sq("cruiser", 5, PARTIAL,     3, 0, 5, (1, 0, 3)), 3),
]
ZHODANI_SCOUT = [
    (_sq("scout", 3, STREAMLINED, 0, 6, 3, None), 1),
    (_sq("scout", 3, STREAMLINED, 0, 7, 7, None), 1),
    (_sq("scout", 2, STREAMLINED, 0, 8, 4, None), 1),
    (_sq("scout", 2, STREAMLINED, 0, 8, 5, None), 1),
    (_sq("scout", 2, STREAMLINED, 0, 4, 4, None), 1),
    (_sq("scout", 2, STREAMLINED, 0, 6, 4, None), 1),
]
ZHODANI_TANKER = [(_sq("tanker", 3, PARTIAL, 0, 0, 4, (0, 0, 2)), 2)]
ZHODANI_ASSAULT = [(_sq("assault", 3, STREAMLINED, 0, 0, 4, (0, 0, 2)), 8)]
ZHODANI_COLONIAL_SQUADRONS = [
    (_sq("battle", 3, UNSTREAMLINED, 4, 3, 7, (2, 1, 3)), 4),
    (_sq("battle", 2, UNSTREAMLINED, 3, 4, 6, (1, 2, 3)), 2),
    (_sq("cruiser", 3, STREAMLINED, 2, 2, 7, (1, 1, 3)), 3),
    (_sq("cruiser", 3, PARTIAL,     2, 2, 6, (1, 1, 3)), 3),
    (_sq("cruiser", 2, STREAMLINED, 0, 2, 6, (0, 1, 3)), 2),
    (_sq("cruiser", 2, PARTIAL,     0, 2, 6, (0, 1, 3)), 1),
    (_sq("cruiser", 2, PARTIAL,     0, 2, 4, (0, 1, 2)), 1),
]

# ---- Sword Worlds --------------------------------------------------------
SWORD_WORLDS_BATTLE = [
    (_sq("battle", 3, UNSTREAMLINED, 3, 0, 5, (1, 0, 3)), 3),
    (_sq("battle", 2, PARTIAL,       4, 2, 8, (2, 1, 4)), 3),
]
SWORD_WORLDS_CRUISER = [
    (_sq("cruiser", 3, PARTIAL, 1, 2, 5, (0, 1, 3)), 3),
    (_sq("cruiser", 2, PARTIAL, 1, 2, 5, (0, 1, 3)), 2),
]
SWORD_WORLDS_ASSAULT = [(_sq("assault", 2, STREAMLINED, 0, 0, 6, (0, 0, 3)), 2)]
SWORD_WORLDS_TANKER = [(_sq("tanker", 3, PARTIAL, 0, 0, 4, (0, 0, 2)), 1)]

# ---- Vargr ---------------------------------------------------------------
VARGR_BATTLE = [(_sq("battle", 3, PARTIAL, 5, 1, 8, (2, 0, 4)), 2)]
VARGR_CRUISER = [(_sq("cruiser", 3, STREAMLINED, 3, 3, 4, (1, 1, 2)), 3)]
VARGR_TANKER = [(_sq("tanker", 3, PARTIAL, 0, 0, 3, (0, 0, 2)), 1)]
VARGR_ASSAULT = [(_sq("assault", 3, STREAMLINED, 0, 0, 6, (0, 0, 3)), 1)]


# --------------------------------------------------------------------------
# troops
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TroopClass:
    size: str            # battalion / regiment / brigade / division / corps / army
    factor: int
    tech_level: int
    armor: bool = False
    elite: bool = False
    jump_troops: bool = False
    mercenary: bool = False
    psionic: bool = False
    name: str = ""


def _tr(size, factor, tl, **kw):
    return TroopClass(size, factor, tl, **kw)


IMPERIAL_TROOPS_5C = [(_tr("army", 500, 15), 8)]
IMPERIAL_TROOPS_1C = [(_tr("corps", 100, 15), 3),
                      (_tr("corps", 100, 14), 2)]
IMPERIAL_TROOPS_20 = [(_tr("division", 20, 15), 5),
                      (_tr("division", 20, 14), 3)]
IMPERIAL_COLONIAL_TROOPS = [
    (_tr("corps", 100, 10), 3), (_tr("corps", 100, 11), 3),
    (_tr("corps", 100, 13), 2), (_tr("brigade", 10, 10), 2),
    (_tr("brigade", 10, 12), 1), (_tr("army", 500, 12), 1),
    (_tr("army", 500, 15), 1), (_tr("division", 20, 10), 1),
]
IMPERIAL_MARINES = [(_tr("division", 20, 15, elite=True), 3),
                    (_tr("brigade", 10, 15, elite=True), 4),
                    (_tr("regiment", 5, 15, elite=True), 1)]
IMPERIAL_HUSCARLES = [(_tr("regiment", 5, 15, elite=True, armor=True), 4),
                      (_tr("battalion", 1, 15, elite=True), 4)]
IMPERIAL_MERCENARIES = [(_tr("battalion", 1, 15, mercenary=True), 2),
                        (_tr("battalion", 1, 16, mercenary=True), 1),
                        (_tr("battalion", 1, 14, mercenary=True), 2),
                        (_tr("battalion", 1, 13, mercenary=True), 2),
                        (_tr("division", 20, 14, mercenary=True,
                             name="PSG"), 1),
                        (_tr("brigade", 10, 12, mercenary=True,
                             name="SPL"), 1)]

ZHODANI_TROOPS_5C = [(_tr("army", 500, 14), 8), (_tr("army", 500, 13), 1)]
ZHODANI_TROOPS_1C = [(_tr("corps", 100, 14), 12)]
ZHODANI_TROOPS_20 = [(_tr("division", 20, 14), 12),
                     (_tr("division", 20, 13), 6)]
ZHODANI_TROOPS_10 = [(_tr("brigade", 10, 13), 1), (_tr("brigade", 10, 12), 1),
                     (_tr("brigade", 10, 11), 1), (_tr("brigade", 10, 10), 1)]
ZHODANI_TROOPS_5 = [(_tr("regiment", 5, 14), 2), (_tr("regiment", 5, 13), 2)]
ZHODANI_COLONIAL_TROOPS = [
    (_tr("corps", 100, 13), 1), (_tr("corps", 100, 12), 1),
    (_tr("division", 20, 8), 1), (_tr("division", 20, 13), 4),
    (_tr("division", 20, 14), 1), (_tr("brigade", 10, 10), 3),
    (_tr("brigade", 10, 12), 2), (_tr("battalion", 2, 10), 2),
    (_tr("battalion", 2, 11), 1), (_tr("battalion", 2, 13), 1),
]
ZHODANI_GUARDS = [(_tr("division", 20, 14, psionic=True, elite=True), 5),
                  (_tr("corps", 100, 14, psionic=True, elite=True), 1)]
FULACIN_BATTALION = _tr("battalion", 1, 14, elite=True, name="Fulacin")

SWORD_WORLDS_TROOPS = [(_tr("corps", 100, 11), 3), (_tr("corps", 100, 10), 3),
                       (_tr("army", 500, 11), 1), (_tr("brigade", 10, 11), 1),
                       (_tr("brigade", 10, 10), 2)]
VARGR_TROOPS = [(_tr("army", 500, 13), 1), (_tr("corps", 100, 12), 3),
                (_tr("division", 20, 13), 3), (_tr("division", 20, 10), 1)]


# --------------------------------------------------------------------------
# admirals -- (precedence, planning factor, tactical factor, name)
# --------------------------------------------------------------------------
IMPERIAL_ADMIRALS = [
    (1, 4, -1, "Grand Admiral"), (2, 1, 1, "Adm. Vasilyev"),
    (3, 4, 0, "Adm. Cheng"), (4, 0, 2, "Adm. Kadlai"),
    (5, 0, 3, "Adm. Arbellatra"), (6, 4, 1, "Adm. Sherlain"),
    (7, 3, 0, "Adm. Lentuli"), (8, 2, 2, "Adm. Ashluda"),
    (9, 4, 3, "Adm. Trellyn"), (10, 4, 1, "Adm. Ivendo"),
    (11, 0, 0, "Adm. Hault-Denz"), (12, 4, 1, "Adm. Norris"),
    (13, 1, -1, "Adm. Sivas"), (14, 4, -1, "Adm. Caranda"),
]
ZHODANI_ADMIRALS = [
    (1, 3, 0, "Demiatl"), (2, 3, 1, "Zdalenstebr"),
    (3, 2, 0, "Kilianstebr"), (4, 3, 1, "Chtaprnentliache"),
    (5, 2, 0, "Polinstliache"), (6, 0, 0, "Tlapiscrfatl"),
    (7, 3, 0, "Qienzhatl"), (8, 1, 0, "Vlanizfatl"),
    (9, 1, 3, "Benshatl"), (10, 0, 3, "Chienjiostebr"),
    (11, 2, -1, "Provincial Governor Shtalialjtes"), (12, 2, 2, "Trodranstebr"),
    (13, 2, 0, "Shanstebr"), (14, 2, 2, "Nivrnditlas"),
]
SWORD_WORLDS_ADMIRALS = [(1, 4, 1, "Rikadattar"), (2, 4, 2, "Eyolfsson"),
                         (3, 2, 2, "Tryggvasson")]
VARGR_ADMIRALS = [(1, 2, 1, "Gireel Pack Leader"), (2, 4, 1, "Uthilh Corsair")]

# --------------------------------------------------------------------------
# fleets
# --------------------------------------------------------------------------
IMPERIAL_FLEETS = ["212th", "213th", "214th", "193rd", "Corridor", "18th",
                   "43rd", "23rd", "100th", "125th", "151st", "187th",
                   "1st Provisional", "TF 17"]
ZHODANI_FLEETS = ["40th", "68th", "65th", "67th", "47th", "28th", "35th",
                  "14th Colonial", "15th Colonial", "16th Colonial",
                  "17th Colonial", "1st Assault", "2nd Assault", "3rd Assault"]
SWORD_WORLDS_FLEETS = ["Joyeuse Fleet", "Gram Fleet"]
VARGR_FLEETS = ["Uthilh Fleet", "Gireel Fleet"]

# --------------------------------------------------------------------------
# replacement point schedules (order of battle charts)
# --------------------------------------------------------------------------
IMPERIAL_REPLACEMENTS = {
    "squadron": {"initial": 0, "turns": [(range(6, 10), 3),
                                         (range(10, 21), 2),
                                         (range(30, 1000), 3)]},
    "troop": {"initial": 3, "turns": [(range(6, 10), 3),
                                      (range(10, 21), 1),
                                      (range(30, 1000), 3)]},
}
ZHODANI_REPLACEMENTS = {
    "squadron": {"initial": 30, "turns": [(range(10, 11), 15),
                                          (range(20, 21), 15),
                                          (range(40, 1000), 2)]},
    "troop": {"initial": 10, "turns": [(range(10, 11), 5),
                                       (range(20, 21), 5),
                                       (range(40, 1000), 1)]},
}

#: Imperial reinforcements table (die roll -> battle, cruiser, fleets)
IMPERIAL_REINFORCEMENT_TABLE = {
    1: (0, 5, 1), 2: (1, 4, 0), 3: (2, 3, 0),
    4: (3, 2, 0), 5: (4, 1, 0), 6: (1, 0, 1),
}


def expand(spec) -> list:
    """``[(class, count), ...] -> [class, class, ...]``."""
    out = []
    for item, count in spec:
        out.extend([item] * count)
    return out
