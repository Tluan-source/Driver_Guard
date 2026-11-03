"""Local debug tooling for the model test console (ui/test-console).

`DebugSession` runs perception + engine with tracing on and turns every tick into a
`LayerDebugFrame` (plain dict) that shows the state of each layer. `server.create_debug_app`
exposes it over a loopback-only WebSocket. Nothing here is used by the production pipeline.
"""
from .session import LAYER_ORDER, DebugSession

__all__ = ["DebugSession", "LAYER_ORDER"]
