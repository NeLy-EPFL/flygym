def find_font_path(family, weight="normal", style="normal"):
    """
    Find the file path of a font given its family, weight, and style.

    Args:
        family: Font family name (e.g., "Arial").
        weight: Font weight (e.g., "normal", "bold").
        style:  Font style (e.g., "normal", "italic").

    Returns:
        The file path of the matching font, or None if not found.
    """
    import matplotlib.font_manager as fm

    font_props = fm.FontProperties(family=family, weight=weight, style=style)
    font_path = fm.findfont(font_props)
    return font_path
