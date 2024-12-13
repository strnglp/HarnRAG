def color_text(text, color):
    """Colors text for terminal output.

    Args:
        text: The text to color.
        color: The color name ('cyan', 'magenta', or other supported colors).

    Returns:
        The colored text string, or the original text if the color is not supported.
    """

    color_codes = {
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "underline": "\033[4m",
        "end": "\033[0m",  # Reset to default color
    }
    return f"{color_codes.get(color, '')}{text}{color_codes['end']}"

