"""Same-fire grouping for annotator smoke alerts.

A fire is not a recurring artefact, so smoke alerts carry no ledger identity —
yet two alerts of one fire, minutes apart on one view, are near-duplicate
footage, and with test open a fire straddling train and test inflates
evaluation. New smoke alerts on the same (organisation, camera, azimuth) view
whose timestamps chain within a window therefore share one split, inheriting
it from already planned smoke when the chain touches any.

The window is generous on purpose: a false merge only makes two fires share a
split, a false split leaks. Matching against the historical pool stays out of
scope, consistent with the import design's accepted residual risks.
"""

from datetime import datetime, timedelta

_TIME_FORMAT = "%Y-%m-%dT%H-%M-%S"

# (timestamp, folder, split) — split is None for a folder not yet planned.
_Event = tuple[datetime, str, str | None]


def view_and_time(folder: str) -> tuple[str, datetime]:
    """Split a folder name into its view (org_camera_azimuth) and time."""
    view, timestamp = folder.rsplit("_", 1)
    return view, datetime.strptime(timestamp, _TIME_FORMAT)


def group_new_smoke(
    new_folders: list[str],
    planned: dict[str, str],
    window_hours: float = 12.0,
) -> list[tuple[list[str], str | None]]:
    """Chain same-view smoke folders within the window into split groups.

    Returns one (new folders of the chain, inherited split) tuple per chain
    holding at least one new folder, ordered by view then time. The inherited
    split is the earliest planned folder's, or None for an entirely new chain.
    Chaining is transitive: consecutive events closer than the window link up,
    however long the chain grows — a 20-hour fire is still one fire.
    """
    window = timedelta(hours=window_hours)
    by_view: dict[str, list[_Event]] = {}
    for folder in new_folders:
        view, at = view_and_time(folder)
        by_view.setdefault(view, []).append((at, folder, None))
    for folder, split in planned.items():
        view, at = view_and_time(folder)
        if view in by_view:
            by_view[view].append((at, folder, split))

    groups: list[tuple[list[str], str | None]] = []
    for view in sorted(by_view):
        chain: list[_Event] = []
        for event in sorted(by_view[view]):
            if chain and event[0] - chain[-1][0] > window:
                _close(chain, groups)
                chain = []
            chain.append(event)
        _close(chain, groups)
    return groups


def _close(chain: list[_Event], groups: list[tuple[list[str], str | None]]) -> None:
    new = [folder for _, folder, split in chain if split is None]
    if not new:
        return
    inherited = next((split for _, _, split in chain if split is not None), None)
    groups.append((new, inherited))
