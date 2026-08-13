# Importing pyro-annotator exports into the sequential dataset

**Status**: design approved, not implemented
**Date**: 2026-08-13
**Scope**: the sequential dataset (`sequential_train_val`, `sequential_test`) feeding the temporal model

## Goal

Feed human-annotated alerts from pyro-annotator into the sequential dataset, adding
smoke sequences and an equal number of false-positive sequences per import, without
data leakage and without the same recurring false positive appearing over and over.

The reference export (2026-08-13, pulled from the test server) holds 767 alerts:
**30 smoke alerts** (35 lanes) and **737 false-positive alerts** (750 lanes), covering
17,502 frames.

## Non-goals

- Changing the 50/50 balance, the two-stage FP selection, or anything about how the
  historical pools were built.
- Cleaning the historical label dirt — tracked separately in
  [#22](https://github.com/pyronear/pyro-dataset/issues/22).
- Lane-level (object-level) dataset semantics. See "Deferred" below.
- Growing the test set. See "Deferred".

## Background: how the pipeline works today

Established by reading the code, and load-bearing for this design:

- **`registry.json` is the single source of truth for splits.** Both
  `scripts/build_wf_yolo_dataset.py:129` and `scripts/build_sequential_dataset.py` read
  the same `seq["split"]`, so the detector dataset and the temporal dataset are aligned
  by construction. Nothing else keeps them in sync.
- **Ingest is append-only.** `scripts/add_data.py` never modifies existing registry
  entries, and `rebalance_minority_splits` swaps only among *new* assignments — never
  touching train or existing entries. A sequence's split is therefore permanent.
- **Splits are stratified per camera.** `compute_new_assignments` groups new folders by
  camera, shuffles with a seed, and assigns each to that camera's most under-represented
  split. Cameras deliberately span splits: 88 wildfire and 124 FP cameras already do.
- **The sequential build balances 50/50 at the sequence level.** `quota = n_wf` per split
  (`build_sequential_dataset.py:149`), with FP sequences chosen by two-stage selection:
  union-find on main-bbox IoU per camera (recurring artefact "atoms"), then KMeans on
  DINOv2 embeddings of the atom representatives.
- **The FP class id is inert.** `fp/selection.py:load_seq_boxes` reads columns 1–5 (+6 as
  confidence) and ignores the class id; `build_fp_yolo_dataset.py:260` writes FP images
  into the detector dataset with `dst_label.touch()`, an empty label. In temporal-model,
  `core/labels.py` keeps `class_id` on the `Detection` but nothing filters on it — a
  sequence is positive or negative according to its `wildfire/` or `fp/` directory
  (`is_wf_sequence`).

Current sizes: 1,552 / 159 / 151 wildfire sequences in train / val / test, matched 1:1 by
FP sequences selected from a 7,154-sequence pool.

## Design

### 1. One alert becomes one sequence folder

`scripts/import_annotator_export.py` reads an export's `manifest.jsonl` and `images/`
and writes staging folders in the layout `add_data.py` already expects:

```
{organisation}_{camera}_{azimuth}_{YYYY-MM-DDTHH-MM-SS}/
├── images/{camera}_{frame-ts}.jpg
├── labels/{camera}_{frame-ts}.txt      # YOLO xywhn, converted from the export's xyxyn
└── meta.json                           # lane-level provenance (see §5)
```

Azimuth is `999` when the alert has none, matching the existing convention.

**All lanes of an alert merge into that single folder**, with their boxes unioned per
frame. This is forced — sibling lanes share camera and start time, so they would produce
identical folder names — and correct: sibling lanes hold byte-identical images (verified:
71 frames in the reference export literally reuse another detection's image, same
sha256). Merging also makes alert-level split binding structural rather than a rule to
enforce.

An alert is classified **wildfire if any lane is smoke**, else **fp**. In the reference
export this is a distinction without a difference — 737 alerts are pure false-positive,
30 are pure smoke, and **none mixes lane kinds**. The rule matters only when a mixed
alert first appears, and it fails safe: putting the frames in `fp/` would train the
temporal model on a sequence that demonstrably contains smoke.

### 2. Label class ids

| box source | class id |
|---|---|
| smoke-lane boxes (human/auto localization) | `0` (`CLASS_ID_SMOKE`) |
| false-positive-lane boxes (engine proposals) | `99` (`CLASS_ID_FALSE_POSITIVE_PROPOSAL`, new) |

Writing `0` on a false-positive box would assert that a building is smoke. `99` is chosen
as a sentinel that collides with none of the ids present in the historical pools
(`0,1,3,4,8,9,10,12,15,18,19`), so annotator-sourced boxes stay greppable and can never be
confused with the dirt in #22. This is safe because no consumer reads the class id (see
Background); it is a correctness-of-record decision, not a functional one.

Labels are written with five columns (`class cx cy w h`). The export carries no detection
confidence — `BoxExport` has `xyxyn`, `smoke_type`, `false_positive_types`, `origin`, and
FP boxes come from the sequence-annotation track, which never stored a score. Five columns
are handled everywhere (`load_seq_boxes` and `core/labels.py` both default confidence to
`1.0`). The cost is confined to `build_fp_yolo_dataset.py`, which ranks hard negatives by
`max_score` and would score these `0.0`; that matters only when the detector starts
consuming annotator data, and is listed under Deferred.

### 3. Selection: all smoke, an equal number of FPs, one per recurring object

**Smoke**: every smoke alert is imported (30 in the reference export). Nothing about them
repeats — 31 groups across 35 lanes.

**False positives**: the smoke count sets the quota, so 30 FP sequences. They are chosen
**one per recurring-object atom, walking down a frequency ranking**, where an atom's
frequency is how many alerts in this export belong to it. No two of the 30 are then the
same artefact.

The expected scale comes from the annotator's own grouping of the same data (a different
implementation of the same idea — see below): it puts the 750 FP lanes into **100 groups
across 42 cameras**, with a heavily skewed frequency distribution — the largest fired 57
times (a road on `sdis-tigery-02`), then a cliff on `brison-03` (32), a building on
`cis-petersbach-01` (30), a light on `morcenx-01` (27). Recomputed atoms will not match
one-for-one, since the thresholds differ, but this is the order of magnitude to expect:
roughly 100 candidates for 30 slots. Taking frequent artefacts first targets the false
positives that actually cost operators attention. **Implementation should report the
recomputed atom count and its agreement with these numbers**, and calibrate the IoU
threshold if they diverge badly — the annotator merges at IoU > 0.3, pyro-dataset's
existing atoms at > 0.7, and for "never the same object twice" the more aggressive merge
is the safer default.

Within an atom the representative is the alert with the most frames, tie-broken by lowest
`platform_alert_id`, so selection is deterministic.

**Atoms are recomputed locally, not imported.** pyro-annotator has its own grouping
(`sequence_group_id`; 783 of 785 exported lanes carry one, over 131 groups), but it is
machine-derived by the same kind of algorithm — median engine bbox, greedy IoU > 0.3, keyed
on `(camera_id, azimuth)` — as pyro-dataset's atoms (NMS top-1 bbox, union-find IoU > 0.7,
per camera, where `camera` already embeds azimuth). Importing the id would couple the
dataset to mutable annotator internals without importing any human judgement. What the
humans did produce — the false-positive **type** — is in the export and rides into
`meta.json`.

Atoms are computed over **annotator-sourced sequences only** (this export plus previously
imported ones), not the historical pool. Residual risk, accepted: 127 exported alerts sit
on 12 cameras the historical pool already uses across all three splits, so one artefact
could exist in both, in different splits. It shrinks as annotator data comes to dominate.

### 4. Splits: train and val only

Annotator sequences are assigned **train/val at 90/10 and never test**.

This is what makes older models comparable across imports. The test split's inputs stay
bit-identical: no new wildfire sequences enter test, so `quota = n_wf` for test is
unchanged; no annotator FP enters the test pool, so its DINOv2 embeddings are unchanged;
KMeans runs with `--random-seed 0`. Same pool, same `k`, same seed produces the same
selected negatives. No lockfile, no pinning, no changes to pyro-dataset internals.

Without this rule the test set is *not* stable: adding data changes `quota`, which changes
`k`, which can change the chosen test negatives wholesale — including for sequences that
were already there and were never touched.

An atom is pinned to exactly one split, permanently. With one sequence per atom this is
free today; it matters when a later import brings more sequences of the same artefact.

### 5. Registry and provenance

Registry entries for annotator sequences carry two extra fields:

```json
{"id": "fp_00007155", "folder": "...", "camera": "...", "split": "train",
 "source": "pyro-annotator", "atom": "<camera>_<azimuth>_<short hash of the atom's representative bbox>"}
```

`source` marks the entries for build-time pinning (§6) and lets a future test refresh
select from atoms that never fed train. `atom` is the ledger: once an atom is seen, its
split is authoritative, so a later import of the same artefact follows it rather than
being re-clustered.

`meta.json` inside each folder keeps what the merge flattens — source alert id, each
lane's kind, smoke types and false-positive types, box origins, atom key. It exists so the
lane-level redesign (Deferred) does not need a re-export, and so a promotion of these
folders elsewhere is a move rather than a re-derivation. Ingest validation checks naming,
`images/`, `labels/` and ≥2 non-empty labels; an extra file is ignored, and the build
copies folders as-is.

### 6. Making the selection stick at build time

Adding sequences to `data/raw/fp/` does not put them in the built dataset — the build
selects `quota = n_wf` per split by KMeans over the whole pool, so the new FPs might simply
not be picked. `build_sequential_dataset.py` is therefore taught to treat entries with
`source: pyro-annotator` as **pinned**: included first, with two-stage selection filling
the remaining quota from the historical pool.

This does not displace anything. The import adds 30 smoke sequences, which raises
`quota = n_wf` by 30 — the pinned FPs consume exactly the slots their own positives
created.

## Invariants

1. The import writes staging folders only. Registry writes go through `add_data.py`;
   nothing ever writes into `data/processed/`. This is what keeps the detector and
   temporal datasets on the same splits.
2. Annotator sequences never receive `split: test`.
3. All sequences of one atom share one split, forever.
4. Smoke boxes are class `0`; false-positive proposal boxes are class `99`.
5. Re-running the import on the same export is a no-op — folder names are stable
   identities and the registry is append-only.

## Verification

- **Idempotency**: run the import twice; the second run adds no folders and no registry
  entries.
- **Leakage**: existing `tests/test_data_leakage.py` covers folder and image overlap across
  sequential splits; extend it with an assertion that no atom appears in two splits.
- **Test-set stability**: build `sequential_test` before and after an import and diff the
  folder listing — they must be identical. Worth checking empirically rather than trusting
  the seed, given the float32 BLAS non-determinism already noted in the two-stage selection.
- **Balance**: after the build, `n_fp == n_wf` per split, unchanged.
- **Spot check**: render overlays for a sample of imported folders (pyro-annotator's
  `make render-overlays` produces the equivalent view from the export side) and confirm
  boxes land on the annotated object.

## Deferred

| Item | Why deferred |
|---|---|
| Growing the test set from annotator data | Needs previously-selected FP test sequences pinned in a lockfile so test can append without re-shuffling. The atom ledger (§5) is what makes it possible later; 70 of 100 atoms stay unused by this import, so the material is not consumed. |
| Lane-level (object-level) dataset semantics | Today a sequence is wildfire or fp as a whole. Lanes are the truer unit, but no alert in the export mixes kinds, so nothing is lost yet. `meta.json` preserves the data for when it is. |
| Confidence column for FP boxes | Requires the annotator export to carry engine confidence, matched back to detections. Only affects the detector's hard-negative ranking. |
| Historical label-id cleanup | [#22](https://github.com/pyronear/pyro-dataset/issues/22). |
| Cross-pool atom matching | Would close the residual camera-overlap risk in §3 once annotator data is no longer marginal. |
