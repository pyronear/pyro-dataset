from pathlib import Path


def is_background(filepath_label: Path) -> bool:
    """
    Is the `filepath_label` a background image - no smokes in it.

    Returns:
        is_background? (bool): whether or not the filepath has a smoke detected in it.
    """
    return (
        filepath_label.exists()
        and filepath_label.is_file()
        and filepath_label.stat().st_size == 0
    )
