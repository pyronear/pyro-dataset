# Runbook: importing a pyro-annotator export

How to take a fresh export from [pyro-annotator](https://github.com/pyronear/pyro-annotator)
— human-annotated alerts — and fold it into the raw pools, the registries, and the
built datasets, without leaking data across splits and without disturbing the
frozen test set.

Run this whenever a meaningful batch of alerts has been finished in the annotator.
Every command below is **idempotent**: if anything is interrupted, re-run the same
command and it continues where it left off.

What the files are (export layout, ledger, plan) is documented in
`data/raw/pyro-annotator/README.md`. Why the flow is shaped this way is in
`docs/specs/2026-08-13-annotator-export-sequential-import-design.md` and
`docs/specs/2026-08-14-annotator-test-growth-design.md`.

## The short version

For operators who have done this before:

```bash
# on a feature branch, from an up-to-date main
dvc pull                                                    # pools, export, embeddings
# refresh data/raw/pyro-annotator/export from pyro-annotator (make export-alerts, step 1), then:
dvc add data/raw/pyro-annotator/export

uv run python scripts/plan_annotator_import.py --dry-run    # read it. really.
uv run python scripts/plan_annotator_import.py
dvc repro materialise_annotator_sequences
uv run python scripts/add_data.py --src data/interim/pyro-annotator/sequences/wildfire \
  --type wildfire --splits-from data/interim/pyro-annotator/sequences/splits.json
uv run python scripts/add_data.py --src data/interim/pyro-annotator/sequences/fp \
  --type fp --splits-from data/interim/pyro-annotator/sequences/splits.json
dvc commit data/raw/wildfire data/raw/fp
dvc repro compute_fp_embeddings
uv run python scripts/freeze_test_selection.py
dvc repro
# review, then commit the state files together (see step 9) and dvc push
```

Everything below explains each step: what it prints, what to check, and what can
go wrong.

## 0. Prerequisites

- A feature branch off an **up-to-date** `main`. The ledger
  (`recurring_objects.json`), the plan (`import_plan.json`) and the test lockfile
  (`sequential_test_lock.json`) are accumulated state living in git — planning
  against stale copies re-decides history. Pull first.
- The data on disk:

  ```bash
  uv sync
  dvc pull   # or targeted: data/raw/pyro-annotator/export (2.2 GB),
             # data/raw/wildfire, data/raw/fp, data/interim/fp_sequence_embeddings,
             # data/processed/sequential_test
  ```

## 1. Refresh the export

The export is pulled by `make export-alerts` in the pyro-annotator repo — the
target lives in `annotation_api/Makefile`, not the repo root. It walks
`GET /api/v1/export/alerts` and downloads images for every **finished** alert.
Images come from presigned S3 URLs, so it runs from any machine that can reach
the API — no need to run it on the VM hosting the annotator.

From a checkout of pyro-annotator, in `annotation_api/`:

```bash
MAIN_ANNOTATION_LOGIN=admin MAIN_ANNOTATION_PASSWORD=... \
make export-alerts \
  REMOTE_API=http://162.19.113.48:5050 \
  OUTPUT_DIR=<path-to-pyro-dataset>/data/raw/pyro-annotator/export
```

- `REMOTE_API` defaults to `https://annotationapi.pyronear.org`; point it at
  whichever deployment is currently production (`162.19.113.48` is the OVH
  annotator VM this runbook's numbers came from). The admin credentials are
  `AUTH_USERNAME` / `AUTH_PASSWORD` in `~/pyro-annotator/.env` on that VM
  (`ssh ubuntu@162.19.113.48`). Pass them as environment variables (or the
  pyro-annotator `.env`) — never commit them.
- Point `OUTPUT_DIR` straight at this repo's
  `data/raw/pyro-annotator/export/`: the pull is idempotent (the manifest is
  rewritten in full, only missing images are downloaded), so writing into the
  DVC-tracked directory is safe.

Then record the new version:

```bash
dvc add data/raw/pyro-annotator/export
```

`git status` now shows `export.dvc` modified — that pointer is what ties this
import to the exact export it read.

Two properties of the export matter for everything below:

- **It is a full re-pull, not a delta.** Every import re-walks the entire
  annotation history; all the deduplication lives downstream, keyed on folder
  names and the ledger.
- **It grows while you look at it.** The annotator is live, so counts drift
  upward between pulls. That is normal.

## 2. Dry-run the plan and actually read it

```bash
uv run python scripts/plan_annotator_import.py --dry-run
```

Nothing is written. The output tells you everything the real run would decide:

Against the current export it prints (your numbers will differ):

```
INFO:root:795 alerts: 58 smoke, 737 false positive; FP quota 0 (58 already ingested)
INFO:root:103 recurring objects, 47 in the hard-negative bucket (score >= 0.5)
INFO:root:selected 0/0 false-positive sequences

DRY RUN — Annotator import plan
  wildfire : 58
  fp       : 0 of a 0 quota, 47 objects fooling
  splits   : train 102, val 14, test 0
  planned  : 0 new folder(s), 116 already planned
```

How to read it:

- **The FP quota is the smoke count minus what earlier imports already
  ingested.** Every new smoke alert is imported; each one buys one
  false-positive slot, so the pools stay balanced. `0 new folder(s)` means this
  export holds nothing that is not already planned — stop here, there is
  nothing to import.
- **`fooling`** counts recurring objects whose best `temporal_model_score` still
  reaches the threshold — false positives the deployed model believes are
  smoke. They are selected first.
- **`splits`** now includes test: new recurring objects and new smoke draw
  80/10/10. Right after the test-growth change, expect a test-heavy first
  import (~15–20 assignments) while the historical 90/10 balance catches up —
  that is by design, not a bug.
- **Warnings are decisions waiting for you.** `changed kind in the export`
  means a registered folder was re-annotated to the opposite kind; the plan
  keeps the registered kind, and actually moving it is
  `scripts/move_fp_to_wildfire.py` / `scripts/move_wf_to_fp.py`, done
  deliberately. `no image on disk` alerts are excluded and cost nothing.
  `same-fire chain ... touches planned folders in ['train', 'val']` flags a
  fire that already straddles splits from before the guard existed.

The dials, when the defaults are not what this import needs:

| flag | what it does |
|---|---|
| `--force-train ro_00042` (repeatable) | force an artefact that is hurting in production into train — refused if it already has sequences in another split |
| `--max-per-object N` | lifetime cap of sequences per recurring object (default 1 — maximum diversity) |
| `--hard-negative-threshold` | score at which an object counts as fooling (default 0.5) |
| `--same-fire-window` | hours within which same-view smoke alerts count as one fire and share a split (default 12) |

## 3. Write the plan

```bash
uv run python scripts/plan_annotator_import.py    # plus any dials from step 2
```

This writes `import_plan.json` and `recurring_objects.json`. Check the diff:

```bash
git diff data/raw/pyro-annotator/
```

What a healthy diff looks like: **new entries only**. Existing plan entries never
change their split; existing ledger objects may gain `seen_alerts` (new
sightings of a known artefact) but their `split` line never moves. If an
existing entry's split changed, stop — that is the one-artefact-in-two-splits
leak the ledger exists to prevent, and something is wrong (most likely a stale
checkout; see step 0).

## 4. Materialise the staging folders

```bash
dvc repro materialise_annotator_sequences
```

Pure copying — a total function of `export/` + `import_plan.json`, no decisions.
It writes `data/interim/pyro-annotator/sequences/{wildfire,fp}/` plus a
`splits.json`, in exactly the layout `add_data.py` expects. The staging
directory is disposable: with the plan in git it can be rebuilt at any time.

## 5. Register both halves

```bash
uv run python scripts/add_data.py --src data/interim/pyro-annotator/sequences/wildfire \
  --type wildfire --splits-from data/interim/pyro-annotator/sequences/splits.json
uv run python scripts/add_data.py --src data/interim/pyro-annotator/sequences/fp \
  --type fp --splits-from data/interim/pyro-annotator/sequences/splits.json
```

`--splits-from` is what makes this different from a platform-loop ingest: splits
come from the file the plan wrote, not from per-camera stratification, so every
sequence lands in the split its recurring object (or same-fire group) was pinned
to. Each command prints a per-camera table and the overall split distribution;
`Skipping N already registered folder(s)` on a re-run is normal.

Then record the grown pools in their DVC pointers:

```bash
dvc commit data/raw/wildfire data/raw/fp
```

## 6. Refresh the embeddings

```bash
dvc repro compute_fp_embeddings
```

Required **before** the freeze whenever the import added FP sequences: the
freeze fills open test slots by clustering over the test pool's DINOv2
embeddings, and stale embeddings make new test FPs invisible to that selection.

## 7. Freeze the test negatives

```bash
uv run python scripts/freeze_test_selection.py --dry-run   # then without --dry-run
```

The FP half of `sequential_test` is not selected at build time — it is recorded
in `data/raw/sequential_test_lock.json`, append-only, and this command is the
only thing that writes it. The output:

```
Test selection freeze
  quota    : 160        # test WF sequences in the registry
  frozen   : 151        # already in the lockfile — never touched
  pinned   : +6 annotator (0 deferred beyond quota)
  filled   : +3 by two-stage selection
```

- `pinned` are annotator test FPs, taken in registry order up to the quota;
  `deferred` ones wait in the pool for a later freeze — nothing is dropped.
- `filled` covers whatever quota the pins did not, selected from the historical
  pool.
- An import that added no test WF sequences opens no slots: the freeze prints
  `+0 / +0` and is a harmless no-op. Run it anyway; it is the check.

Then look at the diff — this is the test set changing, the most consequential
diff of the whole import:

```bash
git diff data/raw/sequential_test_lock.json
```

**Additions at the end of the list only.** Any removed or reordered line means
the append-only guarantee broke; do not commit it.

This applies beyond annotator imports: **any** ingest that adds test WF
sequences (the platform loop included) grows the quota and needs a freeze
before the next build.

## 8. Rebuild and check

```bash
dvc repro
```

- `build_sequential_dataset` copies the lockfile verbatim for test and refuses
  to run with a stale one — if you skipped step 7 it fails with
  `test lockfile has N folders but the quota is M — run
  scripts/freeze_test_selection.py`. That error is the system working.
- `test_data_leakage` runs as a stage: split leakage, recurring-object pinning,
  and lockfile-vs-quota consistency, against the real data.
- The build summary must show `n_fp == n_wf` in every split (50% FP).

Note that a full `dvc repro` needs the export on disk (the materialise stage
reads it); that is why step 0 pulls it.

## 9. Review, commit, push

One import = **one commit** carrying all of its coupled state, so no two of
these files can ever disagree about what was decided:

```bash
git add data/raw/pyro-annotator/export.dvc \
        data/raw/pyro-annotator/import_plan.json \
        data/raw/pyro-annotator/recurring_objects.json \
        data/raw/wildfire.dvc data/raw/fp.dvc \
        data/raw/sequential_test_lock.json \
        dvc.lock
git commit
dvc push
```

Open a PR. The reviewer's fast path: the plan/ledger diffs are additions-only,
the lockfile diff is additions-only, CI's leakage stage is green.

## 10. Release the dataset

A dataset release is a git tag on the merged import commit — the tag pins
`dvc.lock`, which records the exact content hash of every output (see
"Dataset Versioning" in the README). After the PR merges:

```bash
git checkout main && git pull
git tag vX.Y.Z        # `git tag` lists the last one; new data = minor bump
git push origin vX.Y.Z
```

Downstream repos then consume the release with
`dvc import https://github.com/pyronear/pyro-dataset <path> --rev vX.Y.Z`.

## When something goes wrong

| symptom | meaning | fix |
|---|---|---|
| build: `test lockfile has N folders but the quota is M` | an ingest grew the test WF pool and the freeze was not (re)run | `dvc repro compute_fp_embeddings`, then `freeze_test_selection.py` |
| freeze: `only N of M slots filled — the test FP pool is exhausted` | more test smoke than available test negatives | the build will refuse until more FP sequences reach the test split; import more FP data |
| freeze: `lockfile holds N folders, which exceeds the quota` | the lockfile was hand-edited or a registry lost entries | never truncate; find out how it grew — `git log data/raw/sequential_test_lock.json` |
| plan: `--force-train ro_X: refused, it already has sequences in val` | the artefact is already pinned elsewhere; moving it would contaminate evaluation | accept it, or handle the registered sequences deliberately first |
| plan: `changed kind in the export, fp -> wildfire; kept as fp` | re-annotation flipped an alert after its folder was registered | if the flip is right, `scripts/move_fp_to_wildfire.py` |
| add_data: `folder(s) exist on disk but are not in registry` | an earlier run copied folders but died before writing the registry | delete those folders from `data/raw/<type>/data/` and re-run `add_data.py` — a plain re-run skips anything already on disk, registered or not |
| anything interrupted mid-run | — | re-run the same command; plan, add_data and freeze are all idempotent |

Things that must never happen, whatever the shortcut looks like:

- Hand-editing `recurring_objects.json`, `import_plan.json` or
  `sequential_test_lock.json` — the single sanctioned exception is editing an
  object's `"split"` in the ledger **before its first ingest**, which is
  equivalent to `--force-train`.
- Committing a lockfile diff that removes or reorders lines.
- Regenerating any of the three state files from scratch: they are history, and
  losing them re-rolls split decisions that models have already been evaluated
  on.
