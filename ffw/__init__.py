"""Fifth Frontier War -- a playable implementation of the GDW 1981 campaign game.

    from ffw import new_game, play
    from ffw.agents import ScriptedAgent

    state = new_game()
    result = play(state, ScriptedAgent("imperial"), ScriptedAgent("zhodani"))
"""

from .state import GameState, WorldMap, IMPERIAL, ZHODANI          # noqa: F401
from .engine import Engine, new_game, play                          # noqa: F401

__all__ = ["GameState", "WorldMap", "Engine", "new_game", "play",
           "IMPERIAL", "ZHODANI"]
__version__ = "1.0.0"
