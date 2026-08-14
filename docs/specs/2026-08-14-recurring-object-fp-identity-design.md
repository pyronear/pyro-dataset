# Recurring objects as identity for annotator-sourced FP sequences

Design for [#35](https://github.com/pyronear/pyro-dataset/issues/35). Builds on the
annotator import design
([2026-08-13-annotator-export-sequential-import-design.md](2026-08-13-annotator-export-sequential-import-design.md)),
whose ledger this design promotes from a split-pinning device to the selection
identity for its cohort.

## 1. Problem

`compute_fp_embeddings` embeds every FP sequence in the pool, annotator-sourced ones
included. The embedding exists so the two-stage selection can *reconstruct* "same
recurring artefact on the same camera" from geometry and DINOv2 distance. For the
annotator cohort that identity does not need reconstructing:
`data/raw/pyro-annotator/recurring_objects.json` records it authoritatively, one
surrogate id per artefact, maintained across imports.

For exactly this cohort the embedding is also degenerate. Annotator FP labels are five
fields (`99 cx cy w h`) with no confidence column, so `best_scoring_frame` ties on every
frame and the crop that represents the sequence is whichever the unsorted directory
traversal yields first (#29). And the sequential build already pins these sequences
ahead of clustering, then filters them back out of the embedding items — their vectors
are computed and thrown away.

**The principle: embed only what we cannot identify.**

## 2. Decisions

Settled during design review; recorded here so they are not re-litigated silently.

1. **Annotator FP sequences stay in the detector (YOLO) datasets.** Keeping them out
   entirely was an open alternative in #35; it is declined. They contribute background
   images like any pool FP.
2. **The detector pin is one image per recurring object, not per sequence.** The ledger
   id is the identity unit. When `--max-per-object` lifts an artefact above one
   sequence, several sequences of it are distinct temporal negatives for the sequential
   dataset but near-duplicate backgrounds for the detector. Today, with the lifetime
   cap at 1, both granularities produce identical output; they diverge only when caps
   are raised.
3. **The `folder → recurring_object` mapping comes from the ledger**, by inverting each
   entry's `ingested_folders`. The ledger is the authoritative record by construction,
   git-committed and diffable, cumulative across imports. (`import_plan.json` is also
   cumulative and would have worked, but it derives from the ledger; per-folder
   `meta.json` copies live in DVC-tracked data rather than the git source of truth.)
4. **No `recurring_object` field in the registry.** Spec §5 of the import design shows
   registry entries carrying it; the implementation never followed, and with the ledger
   read at build time nothing needs it. Deliberately left unimplemented until something
   does — a backfill would cost an ingest change, a migration, and a `data/raw/fp` DVC
   re-push for no consumer.
5. **The DVC stage layout is unchanged.** True stage-level isolation (annotator imports
   not invalidating the embedding stage) requires the cohort physically outside
   `data/raw/fp` — its own data root and `.dvc` target. That restructure belongs with a
   future revisit of the cohort's place in the pipeline, not here. The win of this
   design is correctness and per-import compute, not rebuild avoidance: any import
   still re-runs the embedding stage, minutes on GPU, but the annotator sequences
   inside that run are no longer cropped and embedded at all.
6. **One recurring object never spans splits.** "One image of the same artefact in each
   split, from different days" was considered as a way to harden evaluation and is
   rejected: the artefact is the unit of leakage. A model trained to stay quiet on an
   object would be evaluated on another sighting of the same object, rewarding
   memorization and inflating exactly the FP metric this cohort exists to measure.
   Harder evaluation comes from whole objects held out (the existing 90/10 object-level
   train/val split of newly minted artefacts) and, later, a test refresh from objects
   that never fed train. Tracking the model on artefacts it *was* trained on ("did we
   fix this nuisance object?") is a legitimate production metric but belongs in a
   separate diagnostic set outside the three splits — follow-up work, out of scope.

## 3. Data flow

The FP pool splits into two cohorts with different identity mechanisms:

- **Pool sequences** (no `source` field): identity must be inferred — geometry atoms,
  DINOv2, KMeans. Unchanged.
- **Annotator sequences** (`source: "pyro-annotator"`): identity is known — the ledger.
  Never embedded, never clustered; selected outright before clustering fills the
  remaining quota.

```
data/raw/fp/registry.json ──partition on source──┬─ pool ──→ compute_fp_embeddings ──→ two_stage_select
                                                 │                                        │ (remaining quota)
                                                 └─ pinned ─┬→ YOLO build: 1 image per   ▼
data/raw/pyro-annotator/recurring_objects.json ─────────────┤   recurring object        outputs
              (ro → ingested_folders, inverted)             └→ sequential build: every
                                                                pinned sequence, as today
```

Two asymmetries are deliberate:

- **Granularity differs per builder** (decision 2): the YOLO build pins one image per
  recurring object; the sequential build pins every sequence, as it already does.
- **Only the YOLO build reads the ledger.** The sequential build takes all pinned
  sequences and needs no identity to do so.

## 4. Component changes

### `src/pyro_dataset/fp/selection.py`

Gains the shared pinning vocabulary, moved verbatim from
`scripts/build_sequential_dataset.py`:

- `ANNOTATOR_SOURCE`
- `partition_pinned(sequences) -> (pinned, rest)`
- `remaining_quota(quota, pinned) -> int`

New:

- `folder_to_recurring_object(ledger: dict, folders: set[str]) -> dict[str, str]` —
  inverts `ingested_folders` restricted to `folders`. Raises `ValueError` if a
  requested folder appears under no object or under more than one. A corrupted ledger
  is a stop-the-build event, not a warning: a silent fallback would quietly
  re-introduce the estimated identity this design removes. Ledger folders outside
  `folders` are ignored — `record_ingested` stamps at planning time, so the ledger may
  list folders that were staged but never registered.

### `scripts/embed_fp_for_selection.py`

Per split, `partition_pinned` on the registry entries; embed only the rest.
`embeddings_meta.json` then contains only sequences whose identity has to be inferred.
Output format unchanged, fewer rows. Docstring updated: the annotator cohort is
identified by the ledger, not embedded.

### `scripts/build_fp_yolo_dataset.py`

New argument `--recurring-objects` (default
`data/raw/pyro-annotator/recurring_objects.json`); `dvc.yaml` passes it and adds the
file as a stage dep.

For train and val: partition pinned; map pinned folders to their recurring object via
`folder_to_recurring_object`; group by object; emit one representative image per
object; then `two_stage_select` fills `remaining_quota(quota, objects)` from the pool
(whose embeddings no longer contain annotator rows).

**Representative rule, deterministic at both levels:** within an object, its
lexicographically first folder; within that folder, the labeled frame with the highest
score, ties broken by filename. Annotator labels are score-less, so in practice this is
the first labeled frame by name — but the rule stays correct if a scored sequence ever
carries a `source`.

Test split: untouched by this design. When it was written, annotator sequences
could not enter test (`assignments_from_splits` refused it); the test-growth
design has since repealed that refusal, so annotator test FPs will eventually
appear in the round-robin pool — accepted, as the test-growth design declares
detector-test behaviour out of its scope.

### `scripts/build_sequential_dataset.py`

Imports the moved helpers from `selection.py`. Drops the `pinned_folders` exclusion
filter over embedding items — annotator sequences no longer appear there, so the
filter's only remaining effect would be to hide a regression in the embedding skip.
The folder-exists check on pinned sequences stays. Behaviour is identical.

## 5. Error handling

| Condition | Behaviour |
|---|---|
| Pinned folder absent from the ledger | Hard error (`ValueError`), build stops |
| Folder under two recurring objects | Hard error (`ValueError`), build stops |
| Pinned folder registered but absent from disk | Warn and skip (both builders, matching today's sequential behaviour) |
| Ledger folder never registered | Ignored — inversion is restricted to pinned registry folders |
| Pinned objects ≥ quota | Remaining quota clamps to 0; clustering selects nothing |

## 6. Testing

Unit tests follow the repo's existing style: pure helpers tested directly, script
wiring exercised by real runs.

- `tests/test_fp_pinning.py` replaces `tests/test_build_sequential_pinning.py`: the
  five existing `partition_pinned` / `remaining_quota` tests, retargeted at
  `pyro_dataset.fp.selection`.
- `folder_to_recurring_object`: normal inversion; folder under no object raises;
  folder under two objects raises; unrequested ledger folders are ignored.
- YOLO representative rule (script loaded via `importlib`, like the pinning tests
  did): one image per object when an object has two folders; lexicographically first
  folder wins; score-less labels give the first labeled frame by name; a scored label
  outranks name order.

Integration verification, in order:

1. `uv run pytest tests/` — all green, including the untouched leakage tests.
2. Real `embed_fp_for_selection.py` run: `n_items` drops by exactly 50 (train) and
   8 (val); test unchanged; pool rows identical for shared sequences.
3. `--dry-run` of both builders: YOLO train/val still hit quota with 50/8 pinned
   reported; sequential counts identical to today's.
4. `dvc repro` of the affected stages, then the `test_data_leakage` stage as the final
   gate.

## 7. Rollout

- One-time cost: the embedding stage re-runs in full (minutes on GPU). Afterwards,
  annotator imports never pay crop preparation or a DINOv2 forward pass for their
  sequences.
- CLAUDE.md's annotator bullet is updated: both builders pin, and
  `compute_fp_embeddings` skips the cohort.
- The import design's §6 gains a pointer to this document.
- Issue #35 gets a closing note recording decisions 2, 3 and the declined follow-ups
  (registry field, per-object diagnostic set) so they are not lost.
- Relationship to #29: the annotator cohort leaves the tie-break-sensitive path
  entirely; #29 stays open for the pool. #33 (test-set reproducibility) is unaffected
  beyond fewer sequences passing through KMeans.
- Interaction with
  [2026-08-14-annotator-test-growth-design.md](2026-08-14-annotator-test-growth-design.md)
  (designed separately, merged since): it opens `split: test` to annotator imports,
  repealing the "test is never assigned" premise this document originally leaned on.
  Its §5 makes surplus annotator test FPs "eligible for two-stage selection", which
  presumes embeddings this design stops producing. How that lands in practice:
  `freeze_test_selection.py` pins annotator test FPs from the registry — no
  embeddings needed — and its clustering fill simply cannot see the surplus, so a
  deferred sequence waits for future slots instead of competing by embedding
  distance. Weaker than §5's letter, consistent with both designs' spirit: exact
  identity is never traded back for estimated identity.
