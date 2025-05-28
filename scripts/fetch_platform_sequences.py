import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import requests

# from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
# load_dotenv(override=True)

print("HELLO WORLD")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")


def get_token(api_url: str, login: str, pwd: str) -> str:
    response = requests.post(
        f"{api_url}/login/creds",
        data={"username": login, "password": pwd},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def api_get(route: str, headers: Dict[str, str]):
    response = requests.get(route, headers=headers)
    try:
        return response.json()
    except:
        raise Exception(f"API Error: {response.status_code} {response.text}")


def download_image(url: str, path: str):
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
    else:
        logging.warning(f"Failed to download image from {url}")


# ----------- Config ------------
# api_url = os.environ.get("API_URL") + "/api/v1"
# superuser_login = os.environ.get("SUPERADMIN_LOGIN")
# superuser_pwd = os.environ.get("SUPERADMIN_PWD")
# -------------------------------

auth_headers = {
    "Authorization": f"Bearer {get_token(api_url, superuser_login, superuser_pwd)}",
    "Content-Type": "application/json",
}


def fetch_sequences_for_date(
    date_str: str,
    api_url: str,
    headers: Dict[str, str],
    output_root: str = "output_sequences",
):
    logging.info(f"Fetching sequences for date: {date_str}")

    # Step 1: Get cameras and build camera_id -> org_id mapping
    camera_list = api_get(f"{api_url}/cameras/", auth_headers)
    # print(f"camera list: {camera_list}")
    camera_map = {}
    for cam in camera_list:
        camera_map[cam["id"]] = cam["organization_id"]

    # Step 2: Get all sequences for the given date
    sequences = api_get(
        f"{api_url}/sequences/all/fromdate?from_date={date_str}&limit=1000&offset=0",
        auth_headers,
    )
    print(f"sequences: {sequences}")

    # Organize directory structure
    for sequence in sequences:
        sequence_id = sequence["id"]
        org_id = camera_map.get(sequence["camera_id"])
        if org_id is None:
            logging.warning(f"Camera ID {sequence['camera_id']} not found in map.")
            continue

        org_dir = os.path.join(output_root, f"sdis_{org_id}")
        os.makedirs(org_dir, exist_ok=True)

        csv_path = os.path.join(org_dir, "sequences.csv")

        # ✅ Load existing sequence IDs from CSV (if exists)
        existing_ids = set()
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path, usecols=["sequence_id"])
                existing_ids = set(existing_df["sequence_id"].astype(int))
            except Exception as e:
                logging.warning(f"Failed to load existing CSV: {csv_path} ({e})")

        if sequence_id in existing_ids:
            logging.info(f"Sequence {sequence_id} already exists in CSV, skipping.")
            continue

        # --- Proceed only if not already stored ---
        seq_dir = os.path.join(org_dir, f"sequence_{sequence['id']}")
        os.makedirs(seq_dir, exist_ok=True)

        csv_path = os.path.join(org_dir, "sequences.csv")

        # Step 3: Get detections for the sequence
        detections = api_get(
            f"{api_url}/sequences/{sequence['id']}/detections", auth_headers
        )

        # Step 4: Save images and metadata
        rows = []
        for det in detections:
            image_url = det["url"]
            image_filename = f"detection_{det['id']}.jpg"
            image_path = os.path.join(seq_dir, image_filename)
            download_image(image_url, image_path)

            rows.append(
                {
                    "sequence_id": sequence["id"],
                    "camera_id": sequence["camera_id"],
                    "organization_id": org_id,
                    "is_wildfire": sequence.get("is_wildfire", None),
                    "started_at": sequence["started_at"],
                    "last_seen_at": sequence["last_seen_at"],
                    "detection_id": det["id"],
                    "image_path": image_path,
                    "created_at": det["created_at"],
                    "azimuth": det["azimuth"],
                    "bucket_key": det["bucket_key"],
                    "bboxes": det["bboxes"],
                }
            )

        # Append to CSV file
        df = pd.DataFrame(rows)
        if not df.empty:
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode="a", index=False, header=False)
            else:
                df.to_csv(csv_path, index=False)

        print(rows)
    logging.info("✅ Done!")


if __name__ == "__main__":
    # start_date = (
    #     sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    # )
    start_date = datetime.now() - timedelta(days=10)
    print(start_date)
    # start_date = .strptime(start_date, "%Y-%m-%d")
    # print(start_date)
    end_date = datetime.now()
    print(end_date)

    headers = {
        "Authorization": f"Bearer {get_token(api_url, superuser_login, superuser_pwd)}",
        "Content-Type": "application/json",
    }

    while start_date <= end_date:
        fetch_sequences_for_date(start_date.strftime("%Y-%m-%d"), api_url, headers)
        start_date += timedelta(days=1)
