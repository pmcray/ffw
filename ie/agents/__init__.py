"""Agents for *Invasion: Earth*."""

from .base import Agent, RandomAgent
from .heuristic import HeuristicAgent, ScriptedAgent
from .legacy import BeachheadAgent

__all__ = ["Agent", "RandomAgent", "HeuristicAgent", "ScriptedAgent",
           "BeachheadAgent"]
