# Importing pyro-annotator exports into the sequential dataset

**Status**: design approved, not implemented
**Date**: 2026-08-13
**Scope**: the sequential dataset (`sequential_train_val`, `sequential_test`) feeding the temporal model

## Goal

Feed human-annotated alerts from pyro-annotator into the sequential dataset, adding
smoke sequences and an equal number of false-positive sequences per import, without
data leakage and without the same recurring false positive appearing over and over.

The reference export (2026-08-13, re-pulled from the test server after
`temporal_model_score` shipped) holds 795 alerts: **58 smoke alerts** (70 lanes) and
**737 false-positive alerts** (750 lanes), covering 18,145 frames. Cross-checked against an
independent SQL re-derivation of the export gate, which returned 796 alerts a minute later —
the box is annotated live, so counts drift upward between any two observations.

That drift is why the FP quota is defined as *"the number of smoke alerts in this export"*
rather than a constant: it was 30 in the morning pull and 58 by the evening.

## Non-goals

- Changing the 50/50 balance, the two-stage FP selection, or anything about how the
  historical pools were built.
- Cleaning the historical label dirt — tracked separately in
  [#22](https://github.com/pyronear/pyro-dataset/issues/22).
- Lane-level (object-level) dataset semantics. See "Deferred" below.
- Growing the test set: v1 never assigns test, so the split is untouched. See "Deferred".

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

### 0. Two halves, split at the state boundary

The import is two commands, because only one half can be a DVC stage
([#26](https://github.com/pyronear/pyro-dataset/issues/26)):

| half | command | reads | writes |
|---|---|---|---|
| **plan** — every decision | `scripts/plan_annotator_import.py` | `export/`, the ledger, the plan | `import_plan.json`, the ledger |
| **materialise** — pure copying | `scripts/materialise_annotator_sequences.py` | `export/`, `import_plan.json` | `sequences/`, `splits.json` |

Plan holds the accumulated state: it clusters recurring objects, ranks them, sets the
quota and assigns splits, and its result depends on every import that came before. A
stage's outputs are regenerated from its declared deps, so declaring the ledger an `out`
would let `dvc repro` rebuild it and destroy the split pinning it exists to provide,
while leaving it undeclared would make the stage lie. Plan therefore stays a manual
command — that is also where the judgement lives (`--dry-run`, `--force-train`,
`--max-per-object`).

Materialise is a total function of `export/` + `import_plan.json` — no ledger, no
randomness, no decisions — so it is correctly cached and reproducible, and runs as the
`materialise_annotator_sequences` stage in `dvc.yaml`.

`import_plan.json` maps each decided folder to its kind, split and recurring object:

```json
{"sdis-91_sdis-tigery-02_285_2026-07-06T04-28-05":
  {"kind": "fp", "split": "train", "recurring_object": "ro_00042"}}
```

Like the ledger it is accumulated, small and diffable, so it is **git-tracked** rather
than DVC-tracked, and committed with the ledger by the same import commit. Because both
outlive any single run, plan re-reads the ledger's split onto every plan entry: the
ledger is authoritative, and a stale entry would be one artefact in two splits.

Alerts with no image on disk are dropped at plan time, not while copying. That keeps
materialisation total, and stops an unusable alert from consuming its recurring object's
one slot.

### 1. One alert becomes one sequence folder

`scripts/materialise_annotator_sequences.py` reads an export's `manifest.jsonl` and
`images/` and writes staging folders in the layout `add_data.py` already expects:

```
{organisation}_{camera}_{azimuth}_{YYYY-MM-DDTHH-MM-SS}/
├── images/{camera}_{frame-ts}.jpg
├── labels/{camera}_{frame-ts}.txt      # YOLO xywhn, converted from the export's xyxyn
└── meta.json                           # provenance + every lane's full track (§5)
```

Azimuth is `999` when the alert has none, matching the existing convention.

**All lanes of an alert merge into that single folder.** This is forced — sibling lanes
share camera and start time, so they would produce identical folder names — and correct:
sibling lanes hold byte-identical images (verified: 109 frames in the reference export
literally reuse another detection's image, same sha256). Merging also makes alert-level
split binding structural rather than a rule to enforce.

An alert is classified **wildfire if any lane is smoke**, else **fp**. In the reference
export this is a distinction without a difference — 737 alerts are pure false-positive,
58 are pure smoke, and **none mixes lane kinds** — re-verified on the re-pulled export. The rule matters only when a mixed
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

**The kind that decides all this is the one the plan pinned (§0), not the export's
current answer.** They agree for every alert whose annotation has not changed. When
re-annotation flips one after its folder was registered, the append-only registry
keeps the folder where it is, so the labels and `meta.json` must stay there too —
a folder filed as a false positive holding class-0 boxes is inverted supervision, and
`build_sequential_dataset.py` copies `labels/` as-is. Planning warns on the flip;
moving the sequence for real is `scripts/move_fp_to_wildfire.py`.

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

**Smoke**: every smoke alert is imported — 58 in the reference export, from 70 lanes.

**False positives**: the smoke count sets the quota, so 58 FP sequences here. They are
chosen **one per recurring object, walking down the ranking below**, where a recurring
object's frequency is how many alerts in this export belong to it. No two of the selected
sequences are then the same artefact.

The expected scale comes from the annotator's own grouping of the same data (a different
implementation of the same idea — see below): it puts the 750 FP lanes into **100 groups
across 42 cameras**, with a heavily skewed frequency distribution — the largest fired 57
times (a road on `sdis-tigery-02`), then a cliff on `brison-03` (32), a building on
`cis-petersbach-01` (30), a light on `morcenx-01` (27). Recomputed recurring objects will not match
one-for-one, since the thresholds differ, but this is the order of magnitude to expect:
roughly 100 candidates for 58 slots. Taking frequent artefacts first targets the false
positives that actually cost operators attention. **Implementation should report the
recomputed recurring object count and its agreement with these numbers**, and calibrate the IoU
threshold if they diverge badly — the annotator merges at IoU > 0.3, pyro-dataset's
pyro-dataset's build-time atoms at > 0.7, and for "never the same object twice" the more aggressive merge
is the safer default.

**Hard negatives rank first.** Frequency answers "how often does this cost an operator";
it does not answer "does the model still fall for it". The second question is answered
directly by `temporal_model_score` — the deployed temporal model's own verdict, carried
**per alert** in the export alongside `temporal_model_version` and `temporal_api_version`.
Measured on the 2026-08-13 export: 665 of 737 FP alerts carry a score (90%), distributed
min 0.000 / p50 0.000 / p90 0.560 / max 0.994, all from one model+api pair (`0.2.0`,
`0.3.1`) so they are mutually comparable. **82 FP alerts score ≥ 0.5** — false positives
the deployed model believes are smoke — spread across 29 cameras, led by `antenna` (20),
`combine_harvester` (17), `other` (16) and `building` (12).

Recurring objects are ranked in two buckets, then by frequency within each:

1. **Fooling** — max `temporal_model_score` across the object's alerts ≥
   `--hard-negative-threshold` (default 0.5), ordered by `seen` descending.
2. **The rest** — ordered by `seen` descending.

Bucketing rather than blending keeps the ordering explainable and robust to score noise, and
degrades gracefully: an object whose alerts are all unscored (10%) simply lands in bucket 2.
**If the export carries no `temporal_model_score` at all, everything lands in bucket 2 and the
ranking is frequency-only** — so the importer runs unchanged against older exports.

Bucket 1 need not fill the quota. Those 82 are *alerts*, not distinct recurring objects, so
after clustering the bucket may hold fewer objects than there are slots; bucket 2 then fills
the remainder. Implementation should log the split between the two, since it is the honest
measure of how much hard-negative material an export actually carries.

Within a recurring object, the chosen alert is the highest-scoring one — the specific
sighting that fools the model most — falling back to frame count when none is scored,
tie-broken by lowest `platform_alert_id`. Selection is deterministic either way.

The score's grain is the alert, not the lane, so for a multi-object alert it cannot be
attributed to a particular lane (23 of 795 alerts). Harmless here: an object's rank uses the
max across its alerts, and FP alerts are overwhelmingly single-lane.

For smoke this ordering is unused, since every smoke alert is imported. Should smoke ever
exceed a quota, the analogous rule inverts: rank by *lowest* score first, those being the
plumes the model currently misses. Worth knowing that today there are none to find — every
one of the 58 exported smoke alerts scores **≥ 0.993**, so the deployed model already gets
all of them right. The smoke half of this import buys coverage and raises the FP quota; the
hard examples are all on the false-positive side.

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
ledger's `ingested_folders` list (§5), not per import.

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

Annotator sequences are assigned **train/val at 90/10, never test**. The test split is out
of scope for this implementation: since `quota` is computed per split and no annotator data
enters the test pool, test is untouched by construction — its sequences, its FP candidates
and its embeddings are all unchanged by an import. Assigning test at all means deciding how
test should grow without silently re-rolling what is already in it, which is a separate
piece of work (see Deferred).

A recurring object is pinned to exactly one split, permanently. With one sequence per recurring object this is
free today; it matters when a later import brings more sequences of the same artefact.

**Known limitation — smoke alerts are not grouped.** Recurring objects exist for false
positives; a fire is not a recurring artefact, so smoke alerts get an independent split.
Two alerts of the *same fire* on one camera can therefore straddle train and val. This is
real in the reference export: `sdis-40_laluque-02_61` produced two smoke folders 147
minutes apart on the same camera and azimuth, one in each. The existing leakage tests do
not catch it — they compare folder names and image stems, which differ — and it matches
what the platform loop already does, so it is accepted rather than solved here. Solving it
means defining "the same fire" (camera + azimuth + a time window), which belongs with the
object-level work in Deferred.

**The split is decided per recurring object, not per camera — annotator imports opt out of
the existing stratification.** `compute_new_assignments` enforces the 80/10/10 ratio
*within each camera*: it groups incoming folders by camera, shuffles them with a seed, and
sends each to whichever split is most under-represented for that camera. That policy is
right for the platform flow — every camera then contributes to evaluation — and it stays
untouched for all other data. But it is per *sequence* and decided by *camera*, so it
directly conflicts with object pinning: two sequences of one artefact would be spread
across splits for the sake of a camera's ratio. It is also the reason 88 wildfire and 124
FP cameras currently span more than one split.

For annotator sequences the split therefore comes from the ledger (§5): a matched object
inherits its recorded split, and a newly minted one is assigned 90/10 train/val by seeded
choice. This requires `add_data.py` to accept **pre-assigned splits** rather than computing
them — see §7. Without that change the ledger is merely advisory and camera stratification
silently wins, breaking the pinning invariant.

**Overrides.** An operator often knows which false positive is hurting in production and
wants it trained on. `--dry-run` prints the ranked recurring objects — id, camera, FP type,
`seen`, proposed split — and `--force-train ro_00042` (repeatable) forces one to train;
hand-editing `"split"` in the ledger before first ingest does the same thing. Forcing to
train can only remove evaluation contamination, never create it, so the only guard needed
is a refusal when that object already has sequences in another split, naming the sequences
that block it. Since annotator objects never enter test, the only possible conflict is with
the 10% val slice. This pairs with `--max-per-object`: the artefact you most want gone is
usually a heavy hitter, so "force to train and give it three sequences" is the real gesture.

### 5. Registry and provenance

Registry entries for annotator sequences carry two extra fields:

```json
{"id": "fp_00007155", "folder": "...", "camera": "...", "split": "train",
 "source": "pyro-annotator", "recurring_object": "ro_00042"}
```

`source` marks the entries for build-time pinning (§6) and lets a future test refresh
select from recurring objects that never fed train.

**The recurring object id is a surrogate key, never derived from content.** Recurring objects live in a ledger,
`data/raw/pyro-annotator/recurring_objects.json`:

```json
{"ro_00042": {"camera": "sdis-tigery-02", "azimuth": 285,
              "bbox_xyxyn": [0.61, 0.47, 0.62, 0.49], "split": "train",
              "seen_alerts": ["pyronear_french:50397", "..."],
              "ingested_folders": ["sdis-91_sdis-tigery-02_285_2026-07-06T04-28-05"]}}
```

Every field is load-bearing. `camera`, `azimuth` and `bbox_xyxyn` are the matching signal;
`split` is what a future sighting inherits. The two lists are deliberately distinct:
`seen_alerts` drives the frequency ranking, `ingested_folders` enforces
`--max-per-object` across imports. One list cannot do both — with a cap above 1, a later
import would have no way to know the object had already contributed.

**Both are identities, not tallies, because the export is a full re-pull**: every import
re-walks the entire history. Counting sightings would inflate an old artefact's frequency
by one whole history per import, biasing the ranking against exactly the new artefacts the
import exists to find. Counting ingested sequences fails more subtly: selection would
re-pick the alert it already staged, that pick would be silently dropped as an existing
folder, and the object could never reach a cap above 1 while burning a quota slot every run.

`bbox_xyxyn` is the **per-coordinate median of the alert's dominant lane** — the lane
contributing the most boxes. Per lane, not across lanes: an alert may hold two spatially
separate artefacts, and a median spanning both lands between them, anchoring the object on
a box that exists in neither. NMS cannot serve here either —
the export carries no confidence, so every box ties and the winner is whichever appeared
first in the manifest, an arbitrary first-frame box. Annotator FP boxes are tiny, so an
unrepresentative one makes the IoU match fragile and splits one artefact into several
objects with different splits. The median is robust to a drifting plume and mirrors how
pyro-annotator derives its own group representative.

The **quota discounts what earlier imports ingested** for the same reason: `smoke` counted
from a full re-pull is cumulative, while already-staged folders are skipped, so a second
import would pay twice for the same smoke and over-supply false positives.

There is deliberately no `first_seen`. Nothing reads it, and it is the weaker version of
something already derivable: each of an object's sequences carries its alert timestamp in
the folder name, so the earliest gives first sighting in alert time rather than import
time.

On each import, a new sequence's main bbox is matched by IoU against the stored
`bbox_xyxyn` of existing recurring objects on the same `(camera, azimuth)`, using the same threshold as
the intra-export clustering. A match joins that recurring object and **inherits its split**; no match
mints the next id.

**The anchor bbox is frozen at mint and never moved.** Updating it to each new
sighting made matching depend on the order and history of the walk: replaying one
export against its own ledger re-assigned alerts to other objects and minted
duplicates, and each duplicate drew a fresh split — the one-artefact-in-two-splits
leakage this ledger exists to prevent. Frozen, `match` is a pure function of the
recorded geometry, so a re-pull is a no-op. Verified on the reference export: 103
recurring objects on the first pass and 103 on a replay against the resulting
ledger.

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

This does not displace anything. The import adds as many smoke sequences as the export
holds (58 here), which raises `quota = n_wf` by the same number — the pinned FPs consume
exactly the slots their own positives created.

Since [2026-08-14-recurring-object-fp-identity-design.md](2026-08-14-recurring-object-fp-identity-design.md),
pinning extends to the YOLO build (one image per recurring object) and the
cohort is no longer embedded at all.

### 7. Changes to existing pyro-dataset code

Everything else is new code. Exactly two existing files change, and both changes are
additive — they alter behaviour only for entries carrying `source: pyro-annotator`, so the
platform flow is bit-for-bit unaffected.

| file | change | why |
|---|---|---|
| `scripts/add_data.py` / `src/pyro_dataset/ingest.py` | accept pre-assigned splits (`--splits-from <file>`, or honour a `split` field on incoming folders) instead of always calling `compute_new_assignments` | §4 — the ledger decides the split per recurring object; without this, per-camera stratification overrides it and pinning breaks |
| `scripts/build_sequential_dataset.py` | include `source: pyro-annotator` FP entries before two-stage selection fills the remaining quota | §6 — otherwise KMeans may simply not pick the recurring objects we deliberately chose |

Both deserve a test: one asserting a pre-assigned split survives ingest untouched, one
asserting a pinned FP entry appears in the built split regardless of clustering.

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
- **Test untouched**: no registry entry gains `split: test`, and no folder appears under
  `sequential_test` that was not there before. Asserted on the registry, not by rebuilding —
  comparing built test sets belongs to the deferred test-growth work.
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
| Growing the test set from annotator data | Out of scope for v1 (§4), which never assigns test. Its own piece of work, because growing test means making it **append-only**: `build_sequential_dataset.py` must record the FP sequences chosen per split in a lockfile and reuse them, selecting only enough new ones to cover a raised quota. Without that, adding wildfire sequences to test raises `quota`, hence `k`, and KMeans re-rolls the historical test negatives — two models scored on different data although nobody touched it. That project also owns the empirical check that a build is reproducible at all (build twice, diff `sequential_test`), given the float32 BLAS non-determinism already noted in the two-stage selection. Nothing is consumed meanwhile: most of the ~100 recurring objects go unused by this import, and the ledger (§5) records which never fed train. |
| Object-level dataset (layout v4) | The real mismatch: pyro-annotator annotates objects, temporal-model *learns* on objects (`build_tubes` emits one tube per sequence), but the dataset in between labels whole sequences via a directory name. Not the binding constraint today — 17 of 767 alerts are multi-object (2.2%), none mixed, and `select_longest_tube` would discard the extra tracks anyway — so a revamp only pays once temporal-model trains on multiple labelled tracks per sequence. That is a cross-repo contract change (`list_sequences`, `is_wf_sequence`, `build_tubes`, `build_sequential_dataset.py`, leakage tests, toy dataset) deserving its own brainstorm; `list_sequences` already documents a versioned "v3.0.0 layout", so a v4 is a legitimate successor. `meta.json` (§5) means the data is already there when it happens. |
| Confidence column for FP boxes | Requires the annotator export to carry engine confidence, matched back to detections. Only affects the detector's hard-negative ranking. |
| Historical label-id cleanup | [#22](https://github.com/pyronear/pyro-dataset/issues/22). |
| Cross-pool recurring object matching | Would close the residual camera-overlap risk in §3 once annotator data is no longer marginal. |
