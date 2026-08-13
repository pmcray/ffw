"""A learned agent: a small numpy MLP value network trained by self-play.

The network scores whole positions.  During play the agent evaluates each
candidate destination by applying a light-weight, cheap "what would the board
look like" transform, blends the network's opinion with the hand-built
features, and picks the best option.  Training lives in ``ffw.training``.
"""

from __future__ import annotations

import json
import math

import numpy as np

from .. import features, hexmap
from ..engine import fleet_jump
from ..state import GameState, IMPERIAL, ZHODANI
from .heuristic import HeuristicAgent, DEFAULT_WEIGHTS


class ValueNetwork:
    """Two-layer perceptron with tanh hidden units and a tanh output.

    Output is the predicted final victory margin scaled to [-1, 1] from the
    Zhodani point of view.  Pure numpy so a trained agent loads anywhere.
    """

    def __init__(self, n_in: int = features.N_FEATURES, hidden: int = 32,
                 seed: int | None = 0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 1.0 / math.sqrt(n_in), (n_in, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0, 1.0 / math.sqrt(hidden), (hidden, 1))
        self.b2 = np.zeros(1)

    # -- inference ---------------------------------------------------------
    def __call__(self, x: np.ndarray) -> float:
        return float(self.forward(np.atleast_2d(x))[0, 0])

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._h = np.tanh(x @ self.w1 + self.b1)
        self._y = np.tanh(self._h @ self.w2 + self.b2)
        return self._y

    # -- training ----------------------------------------------------------
    def train(self, x: np.ndarray, target: np.ndarray, lr: float = 0.02,
              epochs: int = 40, batch: int = 64, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        target = target.reshape(-1, 1)
        loss = 0.0
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch):
                idx = order[start:start + batch]
                xb, yb = x[idx], target[idx]
                out = self.forward(xb)
                err = out - yb
                loss = float(np.mean(err ** 2))
                d_out = 2 * err / len(idx) * (1 - out ** 2)
                d_w2 = self._h.T @ d_out
                d_b2 = d_out.sum(0)
                d_h = (d_out @ self.w2.T) * (1 - self._h ** 2)
                d_w1 = xb.T @ d_h
                d_b1 = d_h.sum(0)
                self.w2 -= lr * d_w2
                self.b2 -= lr * d_b2
                self.w1 -= lr * d_w1
                self.b1 -= lr * d_b1
        return loss

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {k: getattr(self, k).tolist()
                for k in ("w1", "b1", "w2", "b2")}

    @classmethod
    def from_dict(cls, data: dict) -> "ValueNetwork":
        net = cls(seed=None)
        for key in ("w1", "b1", "w2", "b2"):
            setattr(net, key, np.asarray(data[key], dtype=np.float64))
        return net

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path: str) -> "ValueNetwork":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


class NeuralAgent(HeuristicAgent):
    """Heuristic doctrine steered by a learned position evaluator.

    The network cannot be queried for every candidate jump -- rolling the whole
    engine forward would be far too slow -- so it is used as a *strategic
    thermostat*: it reads the position, decides whether the war is going well,
    and modulates the doctrine weights (aggression, risk, concentration)
    accordingly.  Everything the network learns is therefore about pace and
    posture, which is exactly the part of Fifth Frontier War that a fixed
    doctrine gets wrong.
    """

    name = "neural"

    def __init__(self, side: str, network: ValueNetwork | None = None,
                 weights=None, seed: int | None = None,
                 label: str | None = None, gain: float = 1.0):
        super().__init__(side, weights, seed, label)
        self.net = network or ValueNetwork()
        self.gain = gain
        self._base = dict(self.w)
        self._assessment = 0.0

    def begin_game(self, state, side, engine):
        super().begin_game(state, side, engine)
        self.w = dict(self._base)

    def _assess(self, state: GameState) -> float:
        raw = features.extract(state)
        value = self.net(features.perspective(raw, ZHODANI))
        return value if self.side == ZHODANI else -value

    def plot_fleet(self, state, side, fleet, turn, engine):
        self._retune(state)
        return super().plot_fleet(state, side, fleet, turn, engine)

    def move_well_led_fleets(self, state, side, fleets, engine):
        self._retune(state)
        return super().move_well_led_fleets(state, side, fleets, engine)

    def _retune(self, state: GameState) -> None:
        """Losing sides press harder and take more risk; winning sides consolidate."""
        value = self._assess(state)
        self._assessment = value
        press = self.gain * value
        self.w = dict(self._base)
        self.w["risk_tolerance"] = self._base["risk_tolerance"] - press
        self.w["vp_value"] = self._base["vp_value"] * (1.0 - 0.5 * press)
        self.w["invasion"] = self._base["invasion"] * (1.0 - 0.4 * press)
        self.w["garrison_gap"] = self._base["garrison_gap"] * (1.0 + 0.8 * press)
        self.w["home_pull"] = self._base["home_pull"] + 0.6 * press
        self.w["concentration"] = self._base["concentration"] * (1.0 + 0.4 * press)

    def declare_armistice(self, state, engine):
        if self.side != ZHODANI or state.turn < 30:
            return False
        # stop when the network thinks the position has peaked
        return self._assess(state) > 0.45 and state.victory_margin() > 55
