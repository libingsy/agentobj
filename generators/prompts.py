"""Prompt assembly for the two-stage MV-Adapter pipeline.

The base prompt plus the semantic prompt describe geometry.
The style prompt is appended only for the appearance stage.
"""


def geometry_prompt(base: str, semantic: str) -> str:
    base = (base or "").strip().strip(",")
    semantic = (semantic or "").strip().strip(",")
    if base and semantic:
        return f"{base}, {semantic}"
    return base or semantic


def appearance_prompt(base: str, semantic: str, style: str) -> str:
    geo = geometry_prompt(base, semantic)
    style = (style or "").strip().strip(",")
    if geo and style:
        return f"{geo}, {style}"
    return geo or style
