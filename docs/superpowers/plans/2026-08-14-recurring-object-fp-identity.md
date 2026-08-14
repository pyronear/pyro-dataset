# Recurring-Object FP Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Annotator-sourced FP sequences stop being embedded with DINOv2; their identity comes from the recurring-object ledger, and the YOLO build pins one background image per recurring object ahead of clustering.

**Architecture:** The FP pool splits into two cohorts. Pool sequences (no `source`) keep the geometry + DINOv2 + KMeans path unchanged. Annotator sequences (`source: "pyro-annotator"`) are never embedded and never clustered: the sequential build pins every sequence (as today), and the YOLO build pins one deterministic image per recurring object, mapped via `recurring_objects.json`. Shared pinning helpers move into `pyro_dataset.fp.selection`.

**Tech Stack:** Python 3.11+ (uv), pytest, DVC, numpy; no new dependencies.

**Spec:** `docs/specs/2026-08-14-recurring-object-fp-identity-design.md` (read it first; decisions there are settled and not to be re-litigated).

## Global Constraints

- Run everything through uv: `uv run pytest ...`, `uv run python ...`.
- Never commit to `main`; all work happens on `feat/recurring-object-fp-identity`.
- Commit messages: repo style prefixes (`feat:`, `refactor:`, `build:`, `docs:`, `test:`). **No AI attribution — no `Co-Authored-By: Claude` lines, no generated-with footers.**
- The user has waived commit signing for this session: run every `git commit` in this plan with `--no-gpg-sign` (the agent shell has no pinentry TTY).
- Before each commit run `uv run ruff check --fix --select I . && uv run ruff format .` (pre-commit enforces both).
- Never run a bare `dvc repro` before Task 1's baseline snapshot exists — a rebuild overwrites the outputs the comparison needs.
- Baseline artifacts live in `data/interim/issue35_baseline/` (git-ignored, DVC-unknown; survives reboots, deleted in Task 6).
- The annotator cohort at time of writing: 50 train + 8 val sequences, each belonging to a distinct recurring object (lifetime cap is 1). Verify, don't assume — Task 1 records the actual numbers.

---

### Task 1: Baseline snapshot

No code changes. Capture what the pipeline currently produces so Task 5 can compare expected vs. actual. Everything goes to `data/interim/issue35_baseline/`.

**Files:**
- Create: `data/interim/issue35_baseline/` (embeddings copy, output listings, annotator folder lists — not committed)

**Interfaces:**
- Consumes: current `data/interim/fp_sequence_embeddings/`, `data/processed/fp_yolo/`, `data/processed/sequential_train_val/`, `data/processed/sequential_test/`
- Produces: `issue35_baseline/embeddings/<split>/{embeddings_dinov2.npz,embeddings_meta.json}`, `issue35_baseline/fp_yolo_<split>.txt`, `issue35_baseline/seq_<split>.txt`, `issue35_baseline/annotator_folders.json`

- [ ] **Step 1: Confirm the workspace matches the lock**

Run: `dvc status`
Expected: no stage listed as changed (data outputs in sync with `dvc.lock`). If stages report modified deps/outs, STOP and ask the user — the baseline would be meaningless.

- [ ] **Step 2: Snapshot embeddings and output listings**

```bash
BASE=data/interim/issue35_baseline
mkdir -p "$BASE"
cp -r data/interim/fp_sequence_embeddings "$BASE/embeddings"
for s in train val test; do
  ls data/processed/fp_yolo/images/$s | sort > "$BASE/fp_yolo_$s.txt"
done
ls data/processed/sequential_train_val/train/fp | sort > "$BASE/seq_train.txt"
ls data/processed/sequential_train_val/val/fp   | sort > "$BASE/seq_val.txt"
ls data/processed/sequential_test/test/fp       | sort > "$BASE/seq_test.txt"
```

- [ ] **Step 3: Record the annotator cohort and baseline counts**

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

BASE = Path("data/interim/issue35_baseline")
reg = json.loads(Path("data/raw/fp/registry.json").read_text())["sequences"]
ann = {}
for s in reg:
    if s.get("source") == "pyro-annotator":
        ann.setdefault(s["split"], []).append(s["folder"])
(BASE / "annotator_folders.json").write_text(json.dumps(ann, indent=2, sort_keys=True))
print({k: len(v) for k, v in ann.items()})  # expected: {'train': 50, 'val': 8}

for split in ("train", "val", "test"):
    meta = json.loads((BASE / "embeddings" / split / "embeddings_meta.json").read_text())
    folders = {it["sequence_folder"] for it in meta["items"]}
    n_ann = len(folders & set(ann.get(split, [])))
    print(f"{split}: n_items={meta['n_items']}  annotator_rows={n_ann}")
EOF
```

Expected: annotator counts `{'train': 50, 'val': 8}`; `annotator_rows` equals those counts (all annotator folders were embedded in the baseline). Record the printed numbers — Task 5 asserts against them.

- [ ] **Step 4: Sanity-check the pinned baseline**

The baseline sequential FP listings must contain every annotator folder (they were pinned by the old code too):

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

BASE = Path("data/interim/issue35_baseline")
ann = json.loads((BASE / "annotator_folders.json").read_text())
for split in ("train", "val"):
    built = set((BASE / f"seq_{split}.txt").read_text().split())
    missing = set(ann.get(split, [])) - built
    print(split, "missing pinned:", sorted(missing))
    assert not missing
EOF
```

Expected: `missing pinned: []` for both splits. No commit for this task.

---

### Task 2: Shared pinning helpers and ledger inversion in `pyro_dataset.fp.selection`

Move `ANNOTATOR_SOURCE`, `partition_pinned`, `remaining_quota` out of `scripts/build_sequential_dataset.py` into the shared selection module, and add `folder_to_recurring_object`.

**Files:**
- Modify: `src/pyro_dataset/fp/selection.py` (add helpers after the module docstring's imports, before `iou_xyxyn`)
- Modify: `scripts/build_sequential_dataset.py` (replace local helper definitions with imports; lines around 59 and 117–136)
- Create: `tests/test_fp_pinning.py`
- Delete: `tests/test_build_sequential_pinning.py`

**Interfaces:**
- Produces (used by Tasks 3 and 4):
  - `ANNOTATOR_SOURCE: str = "pyro-annotator"`
  - `partition_pinned(sequences: list[dict]) -> tuple[list[dict], list[dict]]`
  - `remaining_quota(quota: int, pinned: Sequence) -> int` (len-based; callers pass lists of dicts or Paths)
  - `folder_to_recurring_object(ledger: dict, folders: set[str]) -> dict[str, str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fp_pinning.py`:

```python
"""Both dataset builds must include annotator-sourced FP sequences.

Adding a sequence to data/raw/fp does not put it in a built dataset: the
builds pick `quota` representatives by clustering the whole pool and may
simply not choose it. Sequences the importer selected deliberately are
pinned, and the helpers implementing that live in pyro_dataset.fp.selection
so the YOLO build, the sequential build and the embedding script share them.
Identity for the pinned cohort comes from the recurring-object ledger.
"""

import pytest

from pyro_dataset.fp.selection import (
    folder_to_recurring_object,
    partition_pinned,
    remaining_quota,
)


def test_partition_pinned_splits_on_source():
    pinned, rest = partition_pinned(
        [
            {"folder": "a", "source": "pyro-annotator"},
            {"folder": "b"},
            {"folder": "c", "source": "platform"},
        ]
    )
    assert [s["folder"] for s in pinned] == ["a"]
    assert [s["folder"] for s in rest] == ["b", "c"]


def test_partition_pinned_handles_an_empty_pool():
    assert partition_pinned([]) == ([], [])


def test_pinned_sequences_consume_quota_before_clustering():
    pinned, _ = partition_pinned(
        [{"folder": f"a{i}", "source": "pyro-annotator"} for i in range(4)]
    )
    assert remaining_quota(quota=10, pinned=pinned) == 6


def test_remaining_quota_never_goes_negative():
    pinned, _ = partition_pinned(
        [{"folder": f"a{i}", "source": "pyro-annotator"} for i in range(12)]
    )
    assert remaining_quota(quota=10, pinned=pinned) == 0


def test_remaining_quota_is_untouched_without_pinned_sequences():
    assert remaining_quota(quota=10, pinned=[]) == 10


def test_folder_to_recurring_object_inverts_ingested_folders():
    ledger = {
        "ro_00001": {"ingested_folders": ["cam-a_1"]},
        "ro_00002": {"ingested_folders": ["cam-b_1", "cam-b_2"]},
    }
    assert folder_to_recurring_object(ledger, {"cam-a_1", "cam-b_2"}) == {
        "cam-a_1": "ro_00001",
        "cam-b_2": "ro_00002",
    }


def test_folder_missing_from_ledger_is_a_hard_error():
    ledger = {"ro_00001": {"ingested_folders": ["cam-a_1"]}}
    with pytest.raises(ValueError, match="cam-x_9"):
        folder_to_recurring_object(ledger, {"cam-x_9"})


def test_folder_under_two_objects_is_a_hard_error():
    ledger = {
        "ro_00001": {"ingested_folders": ["dup_1"]},
        "ro_00002": {"ingested_folders": ["dup_1"]},
    }
    with pytest.raises(ValueError, match="dup_1"):
        folder_to_recurring_object(ledger, {"dup_1"})


def test_unrequested_ledger_folders_are_ignored():
    ledger = {
        "ro_00001": {"ingested_folders": ["staged-never-registered", "cam-a_1"]}
    }
    assert folder_to_recurring_object(ledger, {"cam-a_1"}) == {"cam-a_1": "ro_00001"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_fp_pinning.py -v`
Expected: collection error — `ImportError: cannot import name 'partition_pinned' from 'pyro_dataset.fp.selection'`.

- [ ] **Step 3: Add the helpers to `src/pyro_dataset/fp/selection.py`**

Insert immediately before `def iou_xyxyn(a, b) -> float:` (keep the existing imports; add `from collections.abc import Sequence` to them):

```python
ANNOTATOR_SOURCE = "pyro-annotator"


def partition_pinned(sequences: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split registry entries into pinned (annotator-sourced) and the rest.

    Adding a sequence to data/raw/fp does not put it in a built dataset:
    two-stage selection picks `quota` representatives from the whole pool and
    may not choose it. Annotator false positives were selected deliberately
    upstream — one per recurring object, hard negatives first — so they are
    included before clustering fills what remains, and they are never
    embedded: their identity comes from the recurring-object ledger.
    """
    pinned = [s for s in sequences if s.get("source") == ANNOTATOR_SOURCE]
    rest = [s for s in sequences if s.get("source") != ANNOTATOR_SOURCE]
    return pinned, rest


def remaining_quota(quota: int, pinned: Sequence) -> int:
    """Slots left for clustering once the pinned sequences take theirs."""
    return max(quota - len(pinned), 0)


def folder_to_recurring_object(ledger: dict, folders: set[str]) -> dict[str, str]:
    """Invert the ledger's ingested_folders, restricted to `folders`.

    The ledger may list folders that were planned but never registered
    (record_ingested stamps at planning time), so only requested folders are
    mapped. A requested folder missing from the ledger, or claimed by two
    objects, is a hard error: falling back silently would re-introduce the
    estimated identity this mapping replaces.
    """
    mapping: dict[str, str] = {}
    for ro_id, entry in ledger.items():
        for folder in entry.get("ingested_folders", []):
            if folder not in folders:
                continue
            if folder in mapping:
                raise ValueError(
                    f"{folder}: listed under both {mapping[folder]} and {ro_id}"
                )
            mapping[folder] = ro_id
    missing = folders - mapping.keys()
    if missing:
        raise ValueError(
            f"no recurring object recorded for: {', '.join(sorted(missing))}"
        )
    return mapping
```

- [ ] **Step 4: Point `scripts/build_sequential_dataset.py` at the shared helpers**

Replace its import line:

```python
from pyro_dataset.fp.selection import load_embeddings, two_stage_select
```

with:

```python
from pyro_dataset.fp.selection import (
    load_embeddings,
    partition_pinned,
    remaining_quota,
    two_stage_select,
)
```

and delete the script's own `ANNOTATOR_SOURCE`, `partition_pinned`, and `remaining_quota` definitions (the block between `copy_sequence` and `if __name__ == "__main__":`). Touch nothing else in the script in this task.

- [ ] **Step 5: Delete the superseded test file**

Run: `git rm tests/test_build_sequential_pinning.py`

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_fp_pinning.py -v && uv run pytest tests/`
Expected: all PASS (the full suite proves the sequential script still imports cleanly).

- [ ] **Step 7: Commit**

```bash
uv run ruff check --fix --select I . && uv run ruff format .
git add src/pyro_dataset/fp/selection.py scripts/build_sequential_dataset.py tests/test_fp_pinning.py
git commit -m "refactor: share FP pinning helpers, add recurring-object ledger inversion"
```

---

### Task 3: Stop embedding annotator sequences; drop the sequential build's dead filter

One behavioural unit: the embed script skips the cohort, and the sequential build stops filtering out rows that no longer exist. (Do not split across commits in the other order — removing the filter while annotator rows are still embedded would double-select pinned sequences.)

**Files:**
- Modify: `scripts/embed_fp_for_selection.py` (docstring + `main`)
- Modify: `scripts/build_sequential_dataset.py` (the `items_kept` filtering block, currently around lines 205–223, and its docstring)

**Interfaces:**
- Consumes: `partition_pinned` from Task 2.
- Produces: `data/interim/fp_sequence_embeddings/<split>/embeddings_meta.json` whose `items` never contain annotator sequences. Task 4 and 5 rely on this.

- [ ] **Step 1: Skip the cohort in `scripts/embed_fp_for_selection.py`**

Add the import:

```python
from pyro_dataset.fp.selection import partition_pinned
```

In `main()`, change the per-split loop:

```python
    for split in args.splits:
        seqs = by_split.get(split, [])
        # Annotator-sourced sequences are never embedded: their identity comes
        # from the recurring-object ledger, and the builders pin them outright.
        _, seqs = partition_pinned(seqs)
        if not seqs:
            print(f"[{split}] no sequences; skipping")
            continue
```

Update the module docstring's first paragraph to:

```
Compute one DINOv2-base embedding per FP sequence using the highest-scoring
labeled frame. Used by the FP YOLO dataset build pipeline (val + train).
Annotator-sourced sequences (source: pyro-annotator) are skipped: their
identity is recorded exactly in the recurring-object ledger, so the builders
pin them without embeddings (see
docs/specs/2026-08-14-recurring-object-fp-identity-design.md).
```

- [ ] **Step 2: Drop the dead exclusion filter in `scripts/build_sequential_dataset.py`**

Replace the block that builds `items_kept` (from the comment `# Load DINOv2 embeddings + per-sequence metadata for this split.` down to the `emb = emb[_np.asarray(keep_mask)]` line) with:

```python
        # Load DINOv2 embeddings + per-sequence metadata for this split. The
        # annotator cohort is absent by construction — embed_fp_for_selection
        # skips it — so only folders missing from disk are dropped here.
        emb, items = load_embeddings(embeddings_dir, split)
        keep_mask = []
        items_kept: list[dict] = []
        for it in items:
            ok = (fp_data_dir / it["sequence_folder"]).is_dir()
            keep_mask.append(ok)
            if not ok:
                fp_missing_total += 1
            if ok:
                items_kept.append(it)
        if not all(keep_mask):
            import numpy as _np

            emb = emb[_np.asarray(keep_mask)]
```

Delete the now-unused `pinned_folders = {s["folder"] for s in pinned_seqs}` line above it (keep `pinned_paths`).

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
uv run ruff check --fix --select I . && uv run ruff format .
git add scripts/embed_fp_for_selection.py scripts/build_sequential_dataset.py
git commit -m "feat: stop embedding annotator FP sequences; ledger is their identity"
```

---

### Task 4: One pinned image per recurring object in the YOLO build

**Files:**
- Modify: `scripts/build_fp_yolo_dataset.py`
- Modify: `dvc.yaml` (the `build_fp_yolo_dataset` stage)
- Create: `tests/test_build_fp_pinning.py`

**Interfaces:**
- Consumes: `partition_pinned`, `remaining_quota`, `folder_to_recurring_object` from Task 2; annotator-free embeddings from Task 3.
- Produces (module functions on `scripts/build_fp_yolo_dataset.py`, tested via importlib):
  - `best_pinned_image(seq_path: Path) -> Path | None`
  - `select_pinned_images(pinned_seqs: list[dict], ledger: dict, data_dir: Path) -> list[Path]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_fp_pinning.py`:

```python
"""The YOLO FP build pins one background image per recurring object.

Annotator labels are score-less (five fields, class 99), so every frame ties:
the representative must be chosen by a deterministic rule, not directory
traversal order. Within an object: lexicographically first folder; within a
folder: highest label score, ties broken by filename.
"""

import importlib.util
from pathlib import Path

SCRIPT = Path("scripts/build_fp_yolo_dataset.py")

spec = importlib.util.spec_from_file_location("build_fp_yolo_dataset", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

BOX = "99 0.5 0.5 0.1 0.1\n"
SCORED = "0 0.5 0.5 0.1 0.1 0.42\n"


def make_seq(tmp_path, folder, labels):
    seq = tmp_path / folder
    (seq / "images").mkdir(parents=True)
    (seq / "labels").mkdir()
    for stem, text in labels.items():
        (seq / "images" / f"{stem}.jpg").write_bytes(b"jpg")
        (seq / "labels" / f"{stem}.txt").write_text(text)
    return seq


def test_scoreless_labels_pick_the_first_frame_by_name(tmp_path):
    make_seq(tmp_path, "seq", {"b": BOX, "a": BOX, "c": BOX})
    assert module.best_pinned_image(tmp_path / "seq").name == "a.jpg"


def test_a_scored_label_outranks_name_order(tmp_path):
    make_seq(tmp_path, "seq", {"a": BOX, "z": SCORED})
    assert module.best_pinned_image(tmp_path / "seq").name == "z.jpg"


def test_sequence_without_labeled_frames_yields_none(tmp_path):
    seq = tmp_path / "seq"
    (seq / "images").mkdir(parents=True)
    (seq / "labels").mkdir()
    assert module.best_pinned_image(seq) is None


def test_one_image_per_recurring_object(tmp_path):
    make_seq(tmp_path, "cam_2", {"x": BOX})
    make_seq(tmp_path, "cam_1", {"y": BOX})
    make_seq(tmp_path, "other_1", {"z": BOX})
    ledger = {
        "ro_00001": {"ingested_folders": ["cam_1", "cam_2"]},
        "ro_00002": {"ingested_folders": ["other_1"]},
    }
    pinned = [{"folder": "cam_1"}, {"folder": "cam_2"}, {"folder": "other_1"}]
    images = module.select_pinned_images(pinned, ledger, tmp_path)
    # ro_00001 is represented by its lexicographically first folder, cam_1
    assert [i.name for i in images] == ["y.jpg", "z.jpg"]


def test_object_falls_to_its_next_folder_when_the_first_has_no_labels(tmp_path):
    empty = tmp_path / "cam_1"
    (empty / "images").mkdir(parents=True)
    (empty / "labels").mkdir()
    make_seq(tmp_path, "cam_2", {"x": BOX})
    ledger = {"ro_00001": {"ingested_folders": ["cam_1", "cam_2"]}}
    images = module.select_pinned_images(
        [{"folder": "cam_1"}, {"folder": "cam_2"}], ledger, tmp_path
    )
    assert [i.name for i in images] == ["x.jpg"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_build_fp_pinning.py -v`
Expected: FAIL with `AttributeError: module 'build_fp_yolo_dataset' has no attribute 'best_pinned_image'`.

- [ ] **Step 3: Implement in `scripts/build_fp_yolo_dataset.py`**

Extend the selection import and add `defaultdict`:

```python
from collections import defaultdict

from pyro_dataset.fp.selection import (
    folder_to_recurring_object,
    load_embeddings,
    partition_pinned,
    remaining_quota,
    two_stage_select,
)
```

Add after `round_robin_sample`:

```python
def best_pinned_image(seq_path: Path) -> Path | None:
    """Deterministic representative frame for a pinned sequence.

    Annotator labels carry no score, so every frame ties at 0.0 and the sort
    falls through to the filename. The rule stays correct for scored labels.
    """
    candidates = scored_images(seq_path)
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c[0], c[1].name))
    return candidates[0][1]


def select_pinned_images(
    pinned_seqs: list[dict], ledger: dict, data_dir: Path
) -> list[Path]:
    """One background image per recurring object, ledger-identified.

    Several sequences of one artefact are distinct temporal negatives for the
    sequential build but near-duplicate backgrounds here, so the object — not
    the sequence — is the unit. Representative: lexicographically first
    folder that yields a labeled frame.
    """
    ro_by_folder = folder_to_recurring_object(
        ledger, {s["folder"] for s in pinned_seqs}
    )
    folders_by_ro: dict[str, list[str]] = defaultdict(list)
    for s in pinned_seqs:
        folders_by_ro[ro_by_folder[s["folder"]]].append(s["folder"])
    images: list[Path] = []
    for ro_id in sorted(folders_by_ro):
        image = None
        for folder in sorted(folders_by_ro[ro_id]):
            image = best_pinned_image(data_dir / folder)
            if image is not None:
                break
        if image is None:
            logging.warning(f"{ro_id}: no labeled frame in any pinned folder")
            continue
        images.append(image)
    return images
```

Add the CLI argument in `make_cli_parser`:

```python
    parser.add_argument(
        "--recurring-objects",
        type=Path,
        default=Path("data/raw/pyro-annotator/recurring_objects.json"),
        help="Recurring-object ledger; identity for annotator-sourced (pinned) sequences.",
    )
```

In the `__main__` block, after `sequences = load_registry(registry_path)`, add:

```python
    ledger = json.loads(args["recurring_objects"].read_text())
```

Replace the train/val branch of the selection loop:

```python
        if split in TWO_STAGE_SPLITS:
            # Annotator-sourced sequences are pinned: one image per recurring
            # object, chosen deterministically, ahead of the clustering.
            pinned_seqs, _ = partition_pinned(by_split_seqs[split])
            pinned = select_pinned_images(pinned_seqs, ledger, data_dir)
            emb, items = load_embeddings(embeddings_dir, split)
            selected_idx = two_stage_select(
                items=items,
                embeddings=emb,
                data_dir=data_dir,
                quota=remaining_quota(quota, pinned),
                nms_iou=nms_iou,
                match_iou=match_iou,
                seed=seed,
            )
            selected = pinned + [
                data_dir
                / items[i]["sequence_folder"]
                / "images"
                / items[i]["image_name"]
                for i in selected_idx
            ]
            strategies[split] = f"two_stage+{len(pinned)}_pinned"
```

Update the docstring's "Sampling strategy" bullet for train/val by inserting, before the two-stage description:

```
  • train, val — annotator-sourced sequences first: one image per recurring
      object (identity from data/raw/pyro-annotator/recurring_objects.json,
      deterministic frame pick), then TWO-STAGE clustering fills the rest:
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_build_fp_pinning.py -v && uv run pytest tests/`
Expected: all PASS.

- [ ] **Step 5: Wire the ledger into `dvc.yaml`**

In the `build_fp_yolo_dataset` stage, add to `cmd` (after the `--embeddings-dir` line):

```yaml
      --recurring-objects data/raw/pyro-annotator/recurring_objects.json
```

and to `deps`:

```yaml
      - data/raw/pyro-annotator/recurring_objects.json
```

- [ ] **Step 6: Commit**

```bash
uv run ruff check --fix --select I . && uv run ruff format .
git add scripts/build_fp_yolo_dataset.py dvc.yaml tests/test_build_fp_pinning.py
git commit -m "feat: pin one FP image per recurring object in the YOLO build"
```

---

### Task 5: Rebuild and compare against the baseline

Runs the changed stages for real and checks expected vs. actual against Task 1's snapshot. GPU required for the embedding stage (minutes).

**Files:**
- Modify: `dvc.lock` (via `dvc repro`)
- Uses: `data/interim/issue35_baseline/` from Task 1

**Interfaces:**
- Consumes: everything from Tasks 2–4.
- Produces: rebuilt `data/interim/fp_sequence_embeddings`, `data/processed/fp_yolo`, `data/processed/sequential_*`; a verified `dvc.lock`.

- [ ] **Step 1: Re-run the embedding stage**

Run: `dvc repro compute_fp_embeddings`
Expected: completes without error; per-split logs show fewer crops than the baseline for train/val.

- [ ] **Step 2: Compare embeddings — expected vs. actual**

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

import numpy as np

BASE = Path("data/interim/issue35_baseline")
ann = json.loads((BASE / "annotator_folders.json").read_text())
annotator = {f for folders in ann.values() for f in folders}

for split in ("train", "val", "test"):
    old = json.loads((BASE / "embeddings" / split / "embeddings_meta.json").read_text())
    new_dir = Path("data/interim/fp_sequence_embeddings") / split
    new = json.loads((new_dir / "embeddings_meta.json").read_text())
    old_f = [it["sequence_folder"] for it in old["items"]]
    new_f = [it["sequence_folder"] for it in new["items"]]
    assert set(new_f) == set(old_f) - annotator, f"{split}: unexpected item set"
    old_emb = np.load(BASE / "embeddings" / split / "embeddings_dinov2.npz")["embeddings"]
    new_emb = np.load(new_dir / "embeddings_dinov2.npz")["embeddings"]
    idx_old = {f: i for i, f in enumerate(old_f)}
    cos = [float(old_emb[idx_old[f]] @ new_emb[i]) for i, f in enumerate(new_f)]
    print(
        f"{split}: dropped={len(old_f) - len(new_f)} "
        f"(expected {len(set(old_f) & annotator)})  min_cos={min(cos):.6f}"
    )
EOF
```

Expected: dropped = 50 (train), 8 (val), 0 (test); `min_cos` ≈ 1.0. Vectors are L2-normalized, so the dot product is cosine similarity; GPU nondeterminism can cause tiny drift. If `min_cos` < 0.999, report it to the user before continuing — Step 3 tells you whether any downstream difference comes from drift or from code.

- [ ] **Step 3: Prove the sequential build is behaviour-preserving, drift excluded**

Build "derived" embeddings — the baseline vectors with annotator rows removed — and run the *new* sequential build against them. If the refactor is a no-op, its FP selections must match the baseline outputs exactly, because inputs are then bit-identical to what the old code effectively clustered.

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

import numpy as np

BASE = Path("data/interim/issue35_baseline")
ann = json.loads((BASE / "annotator_folders.json").read_text())
annotator = {f for folders in ann.values() for f in folders}

for split in ("train", "val", "test"):
    src = BASE / "embeddings" / split
    meta = json.loads((src / "embeddings_meta.json").read_text())
    emb = np.load(src / "embeddings_dinov2.npz")["embeddings"]
    keep = [
        i
        for i, it in enumerate(meta["items"])
        if it["sequence_folder"] not in annotator
    ]
    out = BASE / "derived_embeddings" / split
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "embeddings_dinov2.npz", embeddings=emb[keep])
    meta["items"] = [meta["items"][i] for i in keep]
    meta["n_items"] = len(keep)
    (out / "embeddings_meta.json").write_text(json.dumps(meta, indent=2))
    print(split, "derived rows:", len(keep))
EOF

uv run python scripts/build_sequential_dataset.py \
  --embeddings-dir data/interim/issue35_baseline/derived_embeddings \
  --output-train-val data/interim/issue35_baseline/seq_tv \
  --output-test data/interim/issue35_baseline/seq_test

diff <(ls data/interim/issue35_baseline/seq_tv/train/fp | sort) data/interim/issue35_baseline/seq_train.txt
diff <(ls data/interim/issue35_baseline/seq_tv/val/fp | sort) data/interim/issue35_baseline/seq_val.txt
diff <(ls data/interim/issue35_baseline/seq_test/test/fp | sort) data/interim/issue35_baseline/seq_test.txt
rm -rf data/interim/issue35_baseline/seq_tv data/interim/issue35_baseline/seq_test
```

Expected: all three `diff`s empty. Any difference here is a code regression (not drift) — stop and debug before proceeding.

- [ ] **Step 4: Rebuild the downstream stages**

Run: `dvc repro build_fp_yolo_dataset build_sequential_dataset`
Expected: both complete; the FP YOLO summary prints `strategy=two_stage+50_pinned` (train) and `two_stage+8_pinned` (val).

- [ ] **Step 5: Compare the YOLO FP dataset — expected vs. actual**

```bash
uv run python - <<'EOF'
import importlib.util
import json
from pathlib import Path

from pyro_dataset.fp.selection import partition_pinned

spec = importlib.util.spec_from_file_location(
    "bfp", "scripts/build_fp_yolo_dataset.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

BASE = Path("data/interim/issue35_baseline")
reg = json.loads(Path("data/raw/fp/registry.json").read_text())["sequences"]
ledger = json.loads(
    Path("data/raw/pyro-annotator/recurring_objects.json").read_text()
)
for split in ("train", "val", "test"):
    built = {p.name for p in Path(f"data/processed/fp_yolo/images/{split}").iterdir()}
    old = set((BASE / f"fp_yolo_{split}.txt").read_text().split())
    print(f"{split}: n={len(built)} (baseline {len(old)}), churn={len(built ^ old)}")
    if split == "test":
        assert built == old, "test is round-robin and must not change"
        continue
    pinned_seqs, _ = partition_pinned([s for s in reg if s["split"] == split])
    expected = m.select_pinned_images(pinned_seqs, ledger, Path("data/raw/fp/data"))
    missing = [i.name for i in expected if i.name not in built]
    print(f"  pinned objects: {len(expected)}, missing from build: {missing}")
    assert not missing
EOF
```

Expected: test identical to baseline; train has 50 pinned objects and val 8, all present; total counts equal the quotas (same as baseline sizes). Churn in the clustered remainder is expected (smaller pool, re-embedded vectors) — report the number, don't assert on it.

- [ ] **Step 6: Verify the real sequential outputs and run the leakage gate**

```bash
diff <(ls data/processed/sequential_train_val/train/fp | sort) data/interim/issue35_baseline/seq_train.txt
diff <(ls data/processed/sequential_train_val/val/fp | sort) data/interim/issue35_baseline/seq_val.txt
diff <(ls data/processed/sequential_test/test/fp | sort) data/interim/issue35_baseline/seq_test.txt
dvc repro test_data_leakage
```

Expected: diffs empty *if* Step 2 showed no drift; with drift, any changed folders must be explainable as cluster-pick churn (Step 3 already proved the code itself is behaviour-preserving — report changed folder counts to the user either way). Leakage tests PASS.

- [ ] **Step 7: Rebuild the remaining downstream stages and commit the lock**

```bash
dvc repro
git add dvc.lock
git commit -m "build: rebuild on ledger-identified annotator FP selection"
```

Expected: `dvc repro` re-runs merge/toy/viz stages only; `dvc status` clean afterwards. (Data push to the DVC remote is deliberately left to the user.)

---

### Task 6: Documentation, issue note, cleanup

**Files:**
- Modify: `CLAUDE.md` (the annotator bullet)
- Modify: `docs/specs/2026-08-13-annotator-export-sequential-import-design.md` (§6 pointer)
- Delete: `data/interim/issue35_baseline/`

**Interfaces:** none — prose only.

- [ ] **Step 1: Update the CLAUDE.md bullet**

Replace:

```
- **False-positive boxes are class 99**, not `0`: they mark where the detector
  fired, not smoke. `build_sequential_dataset.py` pins annotator-sourced FP
  sequences ahead of its clustering, so the ones chosen upstream actually reach
  the built dataset.
```

with:

```
- **False-positive boxes are class 99**, not `0`: they mark where the detector
  fired, not smoke. Annotator-sourced FP sequences are never embedded — their
  identity comes from `recurring_objects.json` — and both builders pin them
  ahead of clustering: the sequential build takes every sequence, the YOLO
  build one image per recurring object. See
  `docs/specs/2026-08-14-recurring-object-fp-identity-design.md`.
```

- [ ] **Step 2: Add the pointer to the import design's §6**

Append to the end of §6 ("Making the selection stick at build time"), after the paragraph ending "…the pinned FPs consume exactly the slots their own positives created.":

```
Since [2026-08-14-recurring-object-fp-identity-design.md](2026-08-14-recurring-object-fp-identity-design.md),
pinning extends to the YOLO build (one image per recurring object) and the
cohort is no longer embedded at all.
```

- [ ] **Step 3: Draft the issue #35 closing note and get user approval**

Draft (do NOT post without the user's explicit go-ahead — it is an outward-facing action):

```
Implemented on feat/recurring-object-fp-identity. Decisions recorded in
docs/specs/2026-08-14-recurring-object-fp-identity-design.md:
- Annotator FP sequences stay in the detector datasets; the YOLO build pins
  one image per recurring object (ledger-identified), the sequential build
  keeps pinning every sequence.
- The folder → recurring_object mapping is read from recurring_objects.json
  (inverted ingested_folders); the registry field from the import design's §5
  stays deliberately unimplemented until something needs it.
- Declined follow-ups worth their own issues if wanted: a per-object FP
  regression/diagnostic set outside the three splits; physically separating
  the annotator cohort from data/raw/fp for stage-level DVC isolation.
```

Post (after approval): `gh issue comment 35 --repo pyronear/pyro-dataset --body-file <draft>`

- [ ] **Step 4: Remove the baseline snapshot**

Run: `rm -rf data/interim/issue35_baseline`

- [ ] **Step 5: Run the full suite one last time and commit**

```bash
uv run pytest tests/
uv run ruff check --fix --select I . && uv run ruff format .
git add CLAUDE.md docs/specs/2026-08-13-annotator-export-sequential-import-design.md
git commit -m "docs: record recurring-object FP identity in CLAUDE.md and import spec"
```

Expected: all tests PASS.
