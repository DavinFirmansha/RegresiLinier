import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "live_cam",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
)

def live_cam(label="Live Camera", interval_ms=1500, height=480, key=None):
    val = _component_func(label=label, interval_ms=interval_ms, height=height, key=key, default=None)
    return val
