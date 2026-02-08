import os
import streamlit.components.v1 as components

_component_func = components.declare_component(
    "auto_cam",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
)

def auto_cam(label="Capture", interval_ms=300, height=520, key=None):
    """Webcam auto-capture: tahan tombol/spasi -> capture terus-menerus.
    Returns list of base64 data-URL strings, or None."""
    val = _component_func(label=label, interval_ms=interval_ms, height=height, key=key, default=None)
    return val
