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
  union-find on main-bbox IoU per camera, producing clusters the code calls "atoms" and
  its docstring describes as "a recurring artefact at the camera" — the same concept this
  spec calls a **recurring object** — then KMeans on the DINOv2 embeddings of their
  representatives.
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
└── meta.json                           # provenance + every lane's full track (§5)
```

Azimuth is `999` when the alert has none, matching the existing convention.

**All lanes of an alert merge into that single folder.** This is forced — sibling lanes
share camera and start time, so they would produce identical folder names — and correct:
sibling lanes hold byte-identical images (verified: 71 frames in the reference export
literally reuse another detection's image, same sha256). Merging also makes alert-level
split binding structural rather than a rule to enforce.

An alert is classified **wildfire if any lane is smoke**, else **fp**. In the reference
export this is a distinction without a difference — 737 alerts are pure false-positive,
30 are pure smoke, and **none mixes lane kinds**. The rule matters only when a mixed
alert first appears, and it fails safe: putting the frames in `fp/` would train the
temporal model on a sequence that demonstrably contains smoke.

### 1b. Which boxes reach `labels/`

**Only the boxes of the lane kind that decided the folder.** Not the union across kinds.

| alert | folder | `labels/` contains |
|---|---|---|
| any lane is smoke | `wildfire/` | smoke-lane boxes only |
| every lane is FP | `fp/` | FP-lane boxes |

The reason is `select_longest_tube` (temporal-model `core/tubes.py:297`): `build_tubes`
keeps exactly **one** tube per sequence, chosen by `(length, n_observed)` and blind to the
class id. In a mixed alert the FP lane is a recurring artefact present from frame 0, while
the plume usually appears partway in — so the FP tube is longer, wins selection, and the
sequence, labelled positive by its directory, would train the model with the *building* as
its evidence. Exactly inverted supervision, silently.

The same rule applies within a smoke lane: boxes the annotator flagged with a
`false_positive_type` (the export's `_smoke_lane_boxes` sets this) are excluded from
`labels/`. None exist in the reference export — verified zero — but the rule should not
wait for the first one.

Nothing is discarded: every excluded box survives in `meta.json` as part of its lane's
track (§5), so an object-level consumer loses nothing.

Note the consequence for multi-object alerts of a *single* kind (52124's three plumes,
52155, 53360, 57057): all their smoke boxes are written. The detector wants them all, and
the temporal model keeps the longest tube and ignores the rest — its existing behaviour
for any multi-plume sequence, not something this import introduces.

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

Given §1b, class `99` can only ever appear under `fp/`. Since `build_wf_yolo_dataset.py`
copies class ids verbatim into the detector dataset while FP images enter it with empty
labels, the sentinel can never reach detector training targets.

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
**one per recurring object, walking down a frequency ranking**, where a recurring object's
frequency is how many alerts in this export belong to it. No two of the 30 are then the
same artefact.

The expected scale comes from the annotator's own grouping of the same data (a different
implementation of the same idea — see below): it puts the 750 FP lanes into **100 groups
across 42 cameras**, with a heavily skewed frequency distribution — the largest fired 57
times (a road on `sdis-tigery-02`), then a cliff on `brison-03` (32), a building on
`cis-petersbach-01` (30), a light on `morcenx-01` (27). Recomputed recurring objects will not match
one-for-one, since the thresholds differ, but this is the order of magnitude to expect:
roughly 100 candidates for 30 slots. Taking frequent artefacts first targets the false
positives that actually cost operators attention. **Implementation should report the
recomputed recurring object count and its agreement with these numbers**, and calibrate the IoU
threshold if they diverge badly — the annotator merges at IoU > 0.3, pyro-dataset's
pyro-dataset's build-time atoms at > 0.7, and for "never the same object twice" the more aggressive merge
is the safer default.

Within a recurring object, alerts are ordered by frame count descending, tie-broken by lowest
`platform_alert_id`, and the first is taken — so selection is deterministic.

**`--max-per-object` (default 1, range 1–10)** allows a recurring object to contribute more than one
sequence. It changes only this step; the ledger, split inheritance and build-time pinning
are unaffected, and multiple sequences of one recurring object cannot leak because they inherit its
single split. Note that with a uniform walk the cap never binds while recurring objects outnumber the
quota (100 vs 30 here) — it takes effect under frequency-proportional allocation, where
slots are handed out in proportion to `seen` and clamped to the cap. That is the dial
between diversity and frequency realism, and inside a fixed quota the trade is strict:
each extra copy of the road costs one distinct object (a cap of 3 yields roughly 15–18
distinct objects instead of 30). Default 1 — maximum diversity — until training results
justify weighting the head of the distribution. The cap is enforced across imports via the
ledger's `ingested` counter (§5), not per import.

**Recurring objects are recomputed locally, not imported.** pyro-annotator has its own grouping
(`sequence_group_id`; 783 of 785 exported lanes carry one, over 131 groups), but it is
machine-derived by the same kind of algorithm — median engine bbox, greedy IoU > 0.3, keyed
on `(camera_id, azimuth)` — as pyro-dataset's build-time atoms (NMS top-1 bbox, union-find IoU > 0.7,
per camera, where `camera` already embeds azimuth). Importing the id would couple the
dataset to mutable annotator internals without importing any human judgement. What the
humans did produce — the false-positive **type** — is in the export and rides into
`meta.json`.

Recurring objects are computed over **annotator-sourced sequences only** (this export plus previously
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

A recurring object is pinned to exactly one split, permanently. With one sequence per recurring object this is
free today; it matters when a later import brings more sequences of the same artefact.

### 5. Registry and provenance

Registry entries for annotator sequences carry two extra fields:

```json
{"id": "fp_00007155", "folder": "...", "camera": "...", "split": "train",
 "source": "pyro-annotator", "recurring_object": "ro_00042"}
```

`source` marks the entries for build-time pinning (§6) and lets a future test refresh
select from recurring objects that never fed train.

**The recurring object id is a surrogate key, never derived from content.** Recurring objects live in a ledger,
`data/raw/recurring_objects.json`:

```json
{"ro_00042": {"camera": "sdis-tigery-02", "azimuth": 285,
              "bbox_xyxyn": [0.61, 0.47, 0.62, 0.49],
              "split": "train", "seen": 57, "ingested": 1}}
```

Every field is load-bearing. `camera`, `azimuth` and `bbox_xyxyn` are the matching signal;
`split` is what a future sighting inherits. The two counters are deliberately distinct:
`seen` is how many sightings have ever matched this recurring object, accumulating across
imports so an artefact's importance is not recomputed from whichever export happens to be
in hand, and it drives the frequency ranking; `ingested` is how many of its sequences are
actually in the dataset, and it is what enforces `--max-per-object` across imports. A
single count cannot do both — with a cap above 1, a later import would have no way to know
the recurring object had already contributed and would add its quota again.

There is deliberately no `first_seen`. Nothing reads it, and it is the weaker version of
something already derivable: each of an object's sequences carries its alert timestamp in
the folder name, so the earliest gives first sighting in alert time rather than import
time.

On each import, a new sequence's main bbox is matched by IoU against the stored
`bbox_xyxyn` of existing recurring objects on the same `(camera, azimuth)`, using the same threshold as
the intra-export clustering. A match joins that recurring object and **inherits its split**; no match
mints the next id. The stored bbox may be updated as members join.

The key must not be a hash of the representative bbox, which was the first sketch and is
wrong: a recurring object's representative drifts by design — it is the member closest to the centroid,
and the centroid moves as sightings accumulate — so the hash would change and the same
physical artefact would be minted as a new recurring object, free to land in a different split. That is
exactly the leakage the ledger exists to prevent. Identity must live in the id; geometry is
only the matching signal. This mirrors pyro-annotator's own sweep, which matches a new
sequence's median bbox against existing groups keyed on `(camera_id, azimuth)`.

**`meta.json`** inside each folder carries provenance *and* every lane's full track — the
source alert id, the recurring object id, and for each lane its kind, smoke types, false-positive
types, and its per-frame boxes with their origins, including lanes and boxes that §1b
excluded from `labels/`:

```json
{"source_api": "pyronear_french", "platform_alert_id": 57057, "recurring_object": "ro_00042",
 "lanes": [{"sequence_id": 7236, "kind": "smoke", "smoke_types": ["industrial"],
            "false_positive_types": [], "in_labels": true,
            "track": [{"frame": "chateau-eau-milly-la-foret-02_2026-08-05T13-46-08",
                       "xyxyn": [0.72, 0.30, 0.74, 0.32], "origin": "human"}]}]}
```

This is the lossless half of the design. The folder layout stays byte-compatible with what
pyro-dataset and temporal-model expect today, so nothing downstream changes; but the
object-level truth is on disk from this import onward, so the redesign in Deferred needs no
re-export and no re-annotation. Ingest validation checks naming, `images/`, `labels/` and
≥2 non-empty labels; an extra file is ignored, and the build copies folders as-is.

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
3. All sequences of one recurring object share one split, forever. Recurring object identity is the ledger id,
   never a value derived from geometry.
4. Smoke boxes are class `0`; false-positive proposal boxes are class `99`. `labels/`
   holds only the boxes of the lane kind that decided the folder (§1b); everything else
   lives in `meta.json`.
5. Re-running the import on the same export is a no-op — folder names are stable
   identities and the registry is append-only.

## Verification

- **Idempotency**: run the import twice; the second run adds no folders and no registry
  entries.
- **Leakage**: existing `tests/test_data_leakage.py` covers folder and image overlap across
  sequential splits; extend it with an assertion that no recurring object appears in two splits.
- **Test-set stability**: build `sequential_test` before and after an import and diff the
  folder listing — they must be identical. Worth checking empirically rather than trusting
  the seed, given the float32 BLAS non-determinism already noted in the two-stage selection.
- **Balance**: after the build, `n_fp == n_wf` per split, unchanged.
- **Losslessness**: for every imported alert, the lanes and boxes in `meta.json` round-trip
  the export's objects exactly — count of lanes, frames per lane, boxes per frame — so
  what §1b excludes from `labels/` is provably still on disk.
- **Recurring object stability**: re-run the import against an export extended with more alerts; every
  previously seen recurring object keeps its id and its split, and only genuinely new artefacts mint
  new ids.
- **Spot check**: render overlays for a sample of imported folders (pyro-annotator's
  `make render-overlays` produces the equivalent view from the export side) and confirm
  boxes land on the annotated object.

## Deferred

| Item | Why deferred |
|---|---|
| Growing the test set from annotator data | Needs previously-selected FP test sequences pinned in a lockfile so test can append without re-shuffling. The recurring object ledger (§5) is what makes it possible later; 70 of 100 recurring objects stay unused by this import, so the material is not consumed. |
| Object-level dataset (layout v4) | The real mismatch: pyro-annotator annotates objects, temporal-model *learns* on objects (`build_tubes` emits one tube per sequence), but the dataset in between labels whole sequences via a directory name. Not the binding constraint today — 17 of 767 alerts are multi-object (2.2%), none mixed, and `select_longest_tube` would discard the extra tracks anyway — so a revamp only pays once temporal-model trains on multiple labelled tracks per sequence. That is a cross-repo contract change (`list_sequences`, `is_wf_sequence`, `build_tubes`, `build_sequential_dataset.py`, leakage tests, toy dataset) deserving its own brainstorm; `list_sequences` already documents a versioned "v3.0.0 layout", so a v4 is a legitimate successor. `meta.json` (§5) means the data is already there when it happens. |
| Confidence column for FP boxes | Requires the annotator export to carry engine confidence, matched back to detections. Only affects the detector's hard-negative ranking. |
| Historical label-id cleanup | [#22](https://github.com/pyronear/pyro-dataset/issues/22). |
| Cross-pool recurring object matching | Would close the residual camera-overlap risk in §3 once annotator data is no longer marginal. |
