"""Agents that can play Fifth Frontier War."""

from .base import Agent, RandomAgent
from .heuristic import (HeuristicAgent, ScriptedAgent, WEIGHTS,
                        DEFAULT_WEIGHTS, weight_vector, weight_dict)
from .learned import NeuralAgent, ValueNetwork
from .doctrine import (DoctrineAgent, RuleSet, Rule, RuleError,
                       default_rules, DEFAULT_RULES)
from .search import LookaheadAgent
from .human import HumanAgent

#: agent classes available for tournaments, by short name
REGISTRY = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "scripted": ScriptedAgent,
    "lookahead": LookaheadAgent,
    "neural": NeuralAgent,
    "doctrine": DoctrineAgent,
    "human": HumanAgent,
}

__all__ = ["Agent", "RandomAgent", "HeuristicAgent", "ScriptedAgent",
           "LookaheadAgent", "NeuralAgent", "ValueNetwork", "HumanAgent",
           "DoctrineAgent", "RuleSet", "Rule", "RuleError", "default_rules",
           "DEFAULT_RULES",
           "REGISTRY", "WEIGHTS", "DEFAULT_WEIGHTS", "weight_vector",
           "weight_dict"]
