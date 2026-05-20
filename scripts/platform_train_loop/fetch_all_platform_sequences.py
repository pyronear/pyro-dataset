"""
CLI script to fetch ALL sequences from the Pyronear alert API within a date
range, skipping any sequence already referenced in the local
`data/raw/wildfire/registry.json` or `data/raw/fp/registry.json`.

Dedup key is `(camera_name, sequence_started_at_normalized)` — the registry
folders are named `<camera>_<YYYY-MM-DDTHH-mm-ss>`, which we match against
platform sequences using each sequence's `started_at` and resolved camera name.

For each new sequence, only the first `--detections-limit` detections
(default 10, ascending by `created_at`) are downloaded.

Usage:
  python fetch_all_platform_sequences.py \
      --save-dir <path> \
      --date-from <YYYY-MM-DD> \
      [--date-end <YYYY-MM-DD>] \
      [--detections-limit 10] \
      [--wildfire-registry ./data/raw/wildfire/registry.json] \
      [--fp-registry ./data/raw/fp/registry.json] \
      [--loglevel info]

Environment variables required (same as fetch_platform_sequences.py):
  PLATFORM_API_ENDPOINT (str): API url endpoint. eg https://alertapi.pyronear.org
  PLATFORM_LOGIN (str): login
  PLATFORM_PASSWORD (str): password
  PLATFORM_ADMIN_LOGIN (str): admin login (used for /organizations)
  PLATFORM_ADMIN_PASSWORD (str): admin password
"""

import argparse
import concurrent.futures
import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import pyro_dataset.platform.api as api
import pyro_dataset.platform.utils as platform_utils
from pyro_dataset.utils import index_by


# Platform camera names look like `croix-augas-01`, while registry camera
# strings look like `pyronear-sdis-77_croix-augas_265` (`<org>_<camera_short>_<azimuth>`).
# `01` vs `02` are physically distinct cameras at the same site that the
# registry cannot tell apart by name alone — we distinguish them via the
# azimuth field, which both the registry (suffix) and the platform
# (`sequence.azimuth`) carry.

# Page size used when paginating `/sequences/all/fromdate`. The platform API
# may return more sequences for a single day than fit in one page; we keep
# requesting pages until a short page is returned.
_SEQUENCES_PAGE_SIZE = 1000


def valid_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a valid date: {s!r}")


def make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        help="Directory to save the sequences",
        type=Path,
        default=Path("./data/raw/pyronear-platform/sequences/all"),
    )
    parser.add_argument(
        "--date-from",
        help="Date in YYYY-MM-DD format (inclusive)",
        type=valid_date,
        required=True,
    )
    parser.add_argument(
        "--date-end",
        help="Date in YYYY-MM-DD format (exclusive), defaults to today",
        type=valid_date,
        default=datetime.now().date(),
    )
    parser.add_argument(
        "--detections-limit",
        help="Maximum number of detections (= images) to fetch per sequence",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--wildfire-registry",
        help="Path to the wildfire registry.json used to detect already-downloaded sequences.",
        type=Path,
        default=Path("./data/raw/wildfire/registry.json"),
    )
    parser.add_argument(
        "--fp-registry",
        help="Path to the fp registry.json used to detect already-downloaded sequences.",
        type=Path,
        default=Path("./data/raw/fp/registry.json"),
    )
    parser.add_argument(
        "-log",
        "--loglevel",
        default="info",
        help="Provide logging level. Example --loglevel debug, default=info",
    )
    return parser


def validate_parsed_args(args: dict) -> bool:
    if args["date_from"] > args["date_end"]:
        logging.error("Invalid combination of --date-from and --date-end parameters")
        return False
    if args["detections_limit"] <= 0:
        logging.error("--detections-limit must be > 0")
        return False
    return True


def validate_available_env_variables() -> bool:
    required = {
        "PLATFORM_API_ENDPOINT": "eg. https://alertapi.pyronear.org",
        "PLATFORM_LOGIN": "",
        "PLATFORM_PASSWORD": "",
        "PLATFORM_ADMIN_LOGIN": "",
        "PLATFORM_ADMIN_PASSWORD": "",
    }
    missing = [name for name in required if not os.getenv(name)]
    for name in missing:
        hint = f" ({required[name]})" if required[name] else ""
        logging.error(f"{name} is not set{hint}")
    return not missing


def get_dates_within(date_from: date, date_end: date) -> list[date]:
    assert date_from <= date_end, "date_from should be <= date_end"
    result = []
    cursor = date_from
    while cursor < date_end:
        result.append(cursor)
        cursor = cursor + timedelta(days=1)
    return result


def collect_existing_sequence_index(
    registry_paths: list[Path],
) -> dict[tuple[str, int], list[str]]:
    """
    Load each `registry.json` and return an index keyed by
    `(timestamp, azimuth_int)` whose value is the list of registry
    `camera_short` strings sharing that key.

    Each registry entry has a `camera` field of the form
    `<org>_<camera_short>_<azimuth_int>` and a `folder` field of the form
    `<camera>_<YYYY-MM-DDTHH-mm-ss>`. The timestamp is recovered by
    stripping the `<camera>_` prefix from `folder`; `camera_short` and
    `azimuth_int` are recovered via `camera.rsplit("_", 2)`.

    A platform sequence is considered a duplicate when:
      - `(_format_api_datetime_str(started_at), int(azimuth))` is in the index, AND
      - the platform `camera.name` matches one of the registry `camera_short`
        strings for that key (see `_camera_matches`).

    Missing files, JSON parse errors, or unexpected shapes are logged and
    that registry is skipped — they do not abort the whole run.
    """
    index: dict[tuple[str, int], list[str]] = {}
    for path in registry_paths:
        if not path.exists():
            logging.warning(f"Registry not found at {path} — skipping")
            continue
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning(f"Failed to read registry {path}: {exc} — skipping")
            continue
        if not isinstance(data, dict):
            logging.warning(
                f"Registry {path} is not a JSON object — skipping (got {type(data).__name__})"
            )
            continue
        sequences = data.get("sequences", [])
        if not isinstance(sequences, list):
            logging.warning(
                f"Registry {path} has non-list `sequences` field — skipping"
            )
            continue
        loaded = 0
        for entry in sequences:
            if not isinstance(entry, dict):
                continue
            camera = entry.get("camera")
            folder = entry.get("folder")
            if not isinstance(camera, str) or not isinstance(folder, str):
                continue
            parts = camera.rsplit("_", 2)
            if len(parts) != 3:
                logging.debug(
                    f"Registry camera {camera!r} does not fit <org>_<cam>_<az>; skipping"
                )
                continue
            _org, camera_short, azimuth_str = parts
            try:
                azimuth_int = int(azimuth_str)
            except ValueError:
                logging.debug(
                    f"Registry camera {camera!r} has non-int azimuth {azimuth_str!r}; skipping"
                )
                continue
            prefix = f"{camera}_"
            if not folder.startswith(prefix):
                logging.debug(
                    f"Registry entry folder {folder!r} does not start with camera prefix {prefix!r}; skipping"
                )
                continue
            timestamp = folder[len(prefix):]
            index.setdefault((timestamp, azimuth_int), []).append(camera_short)
            loaded += 1
        logging.info(f"Loaded {loaded} sequence entry/entries from {path}")
    return index


def _camera_matches(platform_name: str, registry_camera_short: str) -> bool:
    """
    Return whether a platform `camera.name` (e.g. `croix-augas-01`) refers
    to the same physical site as a registry `camera_short` (e.g. `croix-augas`).

    True when:
      - the strings are equal, OR
      - the platform name extends the registry short name with a
        `-<digits>` suffix (the per-site camera index `01`, `02`, ...).

    Note: the registry cannot itself distinguish `croix-augas-01` from
    `croix-augas-02`; those are disambiguated by the azimuth carried in the
    index key, not by this function.
    """
    if platform_name == registry_camera_short:
        return True
    if not platform_name.startswith(registry_camera_short + "-"):
        return False
    suffix = platform_name[len(registry_camera_short) + 1:]
    return suffix.isdigit()


def _fetch_sequences_for_date(
    api_endpoint: str, mdate: date, access_token: str
) -> list:
    """
    Page through `/sequences/all/fromdate` for `mdate`. Returns the full
    list — keeps requesting pages until a short page is returned.
    """
    all_sequences: list = []
    offset = 0
    while True:
        page = api.list_sequences_for_date(
            api_endpoint=api_endpoint,
            date=mdate,
            limit=_SEQUENCES_PAGE_SIZE,
            offset=offset,
            access_token=access_token,
        )
        if not isinstance(page, list):
            logging.warning(
                f"Unexpected response paginating {mdate} at offset {offset}: "
                f"{type(page).__name__}"
            )
            break
        all_sequences.extend(page)
        if len(page) < _SEQUENCES_PAGE_SIZE:
            break
        offset += _SEQUENCES_PAGE_SIZE
    return all_sequences


def _process_sequence(
    api_endpoint: str,
    sequence: dict,
    detections_limit: int,
    indexed_cameras: dict,
    indexed_organizations: dict,
    access_token: str,
) -> list[dict]:
    """
    Fetch the first `detections_limit` detections (ascending by created_at)
    for `sequence` and convert them into records ready for `process_dataframe`.
    """
    detections = api.list_sequence_detections(
        api_endpoint=api_endpoint,
        sequence_id=sequence["id"],
        access_token=access_token,
        limit=detections_limit,
        desc=False,
    )
    records = []
    camera = indexed_cameras.get(sequence["camera_id"])
    if camera is None:
        logging.warning(
            f"Skipping sequence {sequence['id']}: camera {sequence['camera_id']} not found"
        )
        return records
    organization = indexed_organizations.get(camera["organization_id"])
    if organization is None:
        logging.warning(
            f"Skipping sequence {sequence['id']}: organization {camera['organization_id']} not found"
        )
        return records
    for detection in detections:
        records.append(
            platform_utils.to_record(
                detection=detection,
                camera=camera,
                organization=organization,
                sequence=sequence,
            )
        )
    return records


def fetch_new_sequences_within(
    date_from: date,
    date_end: date,
    detections_limit: int,
    api_endpoint: str,
    access_token: str,
    access_token_admin: str,
    existing_index: dict[tuple[str, int], list[str]],
) -> pd.DataFrame:
    cameras = api.list_cameras(api_endpoint=api_endpoint, access_token=access_token)
    indexed_cameras = index_by(cameras, key="id")
    organizations = api.list_organizations(
        api_endpoint=api_endpoint, access_token=access_token_admin
    )
    indexed_organizations = index_by(organizations, key="id")

    dates = get_dates_within(date_from=date_from, date_end=date_end)
    logging.info(
        f"Listing sequences across {len(dates)} day(s): "
        f"{date_from:%Y-%m-%d} → {date_end:%Y-%m-%d}"
    )

    sequences: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_date = {
            executor.submit(
                _fetch_sequences_for_date, api_endpoint, mdate, access_token
            ): mdate
            for mdate in dates
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_date),
            total=len(future_to_date),
            desc="Listing sequences",
        ):
            sequences.extend(future.result())

    logging.info(f"API returned {len(sequences)} sequences")

    new_sequences: list[dict] = []
    skipped_existing = 0
    skipped_missing_camera = 0
    skipped_malformed = 0
    for s in sequences:
        camera_id = s.get("camera_id")
        started_at = s.get("started_at")
        azimuth = s.get("azimuth")
        if (
            camera_id is None
            or not isinstance(started_at, str)
            or not isinstance(azimuth, (int, float))
        ):
            skipped_malformed += 1
            logging.debug(
                f"Skipping malformed sequence (id={s.get('id')!r}): "
                f"camera_id={camera_id!r}, started_at={started_at!r}, azimuth={azimuth!r}"
            )
            continue
        camera = indexed_cameras.get(camera_id)
        if camera is None:
            skipped_missing_camera += 1
            continue
        timestamp = platform_utils._format_api_datetime_str(started_at)
        key = (timestamp, int(azimuth))
        registry_camera_shorts = existing_index.get(key, [])
        if registry_camera_shorts:
            platform_name = camera.get("name", "")
            if any(
                _camera_matches(platform_name, registry_cam)
                for registry_cam in registry_camera_shorts
            ):
                skipped_existing += 1
                continue
        new_sequences.append(s)
    msg_parts = [f"Skipping {skipped_existing} sequence(s) already in registries"]
    if skipped_missing_camera:
        msg_parts.append(f"{skipped_missing_camera} with unresolved camera")
    if skipped_malformed:
        msg_parts.append(f"{skipped_malformed} malformed")
    msg_parts.append(f"downloading {len(new_sequences)} new sequence(s)")
    logging.info("; ".join(msg_parts))

    if not new_sequences:
        return pd.DataFrame()

    records: list[dict] = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_sequence = {
            executor.submit(
                _process_sequence,
                api_endpoint,
                sequence,
                detections_limit,
                indexed_cameras,
                indexed_organizations,
                access_token,
            ): sequence
            for sequence in new_sequences
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_to_sequence),
            total=len(future_to_sequence),
            desc="Fetching detections",
        ):
            records.extend(future.result())

    logging.info(
        f"Collected {len(records)} detection record(s) "
        f"for {len(new_sequences)} new sequence(s)"
    )
    return pd.DataFrame(records)


if __name__ == "__main__":
    cli_parser = make_cli_parser()
    args = vars(cli_parser.parse_args())
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=args["loglevel"].upper())

    if not validate_parsed_args(args):
        exit(1)
    if not validate_available_env_variables():
        exit(1)

    logger.info(args)
    platform_api_endpoint = os.getenv("PLATFORM_API_ENDPOINT")
    platform_login = os.getenv("PLATFORM_LOGIN")
    platform_password = os.getenv("PLATFORM_PASSWORD")
    platform_admin_login = os.getenv("PLATFORM_ADMIN_LOGIN")
    platform_admin_password = os.getenv("PLATFORM_ADMIN_PASSWORD")
    if (
        not platform_api_endpoint
        or not platform_login
        or not platform_password
        or not platform_admin_login
        or not platform_admin_password
    ):
        logger.error("Missing platform credentials...")
        exit(1)

    save_dir: Path = args["save_dir"]
    date_from: date = args["date_from"]
    date_end: date = args["date_end"]
    detections_limit: int = args["detections_limit"]
    wildfire_registry: Path = args["wildfire_registry"]
    fp_registry: Path = args["fp_registry"]

    logger.info(
        f"Fetching sequences from {date_from:%Y-%m-%d} until {date_end:%Y-%m-%d} "
        f"(first {detections_limit} detection(s)/sequence), saving to {save_dir} "
        f"via {platform_api_endpoint}/api/v1"
    )

    logger.info(
        f"Loading existing sequences from registries: {wildfire_registry}, {fp_registry}"
    )
    existing_index = collect_existing_sequence_index(
        [wildfire_registry, fp_registry]
    )
    total_registry_entries = sum(len(v) for v in existing_index.values())
    logger.info(
        f"Loaded {total_registry_entries} registry entry/entries "
        f"across {len(existing_index)} distinct timestamp(s)"
    )

    logger.info("Fetching access tokens...")
    access_token = api.get_api_access_token(
        api_endpoint=platform_api_endpoint,
        username=platform_login,
        password=platform_password,
    )
    access_token_admin = api.get_api_access_token(
        api_endpoint=platform_api_endpoint,
        username=platform_admin_login,
        password=platform_admin_password,
    )
    logger.info("Access tokens obtained ✔️")

    df = fetch_new_sequences_within(
        date_from=date_from,
        date_end=date_end,
        detections_limit=detections_limit,
        api_endpoint=platform_api_endpoint,
        access_token=access_token,
        access_token_admin=access_token_admin,
        existing_index=existing_index,
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.info("No new sequences to download — exiting")
    else:
        filepath_api_results_csv = save_dir / "api_results.csv"
        logger.info(f"Saving API results to {filepath_api_results_csv}")
        platform_utils.append_dataframe_to_csv(
            df=df, filepath_csv=filepath_api_results_csv
        )

        logger.info("Downloading images and writing label files...")
        platform_utils.process_dataframe(df=df, save_dir=save_dir)

    args_content = {
        "date-from": str(date_from),
        "date-end": str(date_end),
        "date-now": datetime.now().strftime("%Y-%m-%d"),
        "save-dir": str(save_dir),
        "wildfire-registry": str(wildfire_registry),
        "fp-registry": str(fp_registry),
        "detections-limit": detections_limit,
        "platform-login": platform_login,
        "platform-admin-login": platform_admin_login,
        "registry-entries-loaded": sum(len(v) for v in existing_index.values()),
        "downloaded-sequences": int(df["sequence_id"].nunique()) if not df.empty else 0,
    }
    filepath_args_yaml = save_dir / "args.yaml"
    logger.info(f"Appending run metadata to {filepath_args_yaml}")
    platform_utils.append_yaml_run(filepath=filepath_args_yaml, data=args_content)
    logger.info("Done ✅")
