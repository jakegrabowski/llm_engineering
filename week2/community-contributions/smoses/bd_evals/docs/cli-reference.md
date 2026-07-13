# CLI Reference

**Status: All advertised commands are wired. Live stage commands require exact
call-count approval and preflight before dataset creation.**

The executable is `rag-evals`.

## Global Options

| Option | Description |
|---|---|
| `--verbose`, `-v` | Enable verbose (DEBUG) logging. |
| `--help` | Show help and exit. |

## Commands

| Command | Status | API calls | Description |
|---|---|---:|---|
| `validate DEFINITION` | implemented | 0 | Resolve and validate configuration/templates |
| `plan DEFINITION` | implemented | 0 | Show matrix, exclusions, IDs, and call counts |
| `run-retrieval DEFINITION` | implemented | `/retrieve` | Build/resume retrieval dataset |
| `judge-retrieval DEFINITION` | implemented | `/ask` | Judge saved retrieval artifacts |
| `generate-answers DEFINITION` | implemented | `/ask` | Generate answers from frozen context |
| `judge-answers DEFINITION` | implemented | `/ask` | Judge saved answer artifacts |
| `report DEFINITION` | implemented | 0 | Produce CSV and Markdown reports |
| `compare DEFINITION` | implemented | 0 | Compare selected datasets |
| `status TYPE DATASET_ID` | implemented | 0 | Inspect dataset lifecycle and work counts |
| `resume DEFINITION` | implemented | varies | Resume eligible missing work |
| `seal TYPE DATASET_ID` | implemented | 0 | Make a complete dataset immutable |
| `clean-dataset TYPE DATASET_ID` | implemented | 0 | Plan or trash an eligible dataset |

Malformed definitions use the normal configuration error path; no command uses a
not-implemented sentinel.

## Implemented Command Details

### validate

```bash
rag-evals validate DEFINITION [--format text|json]
```

Resolves all source references, validates Pydantic schemas, applies selection
rules, and reports placeholder warnings. Outputs definition hash, input list
with content hashes, and validation status.

### plan

```bash
rag-evals plan DEFINITION [--format text|json]
```

Generates the deterministic matrix, prints dimensions, selected IDs, exclusions,
per-row artifact IDs, total call count, and the required `--approve-call-count`
value.

### run-retrieval

```bash
rag-evals run-retrieval DEFINITION \
  [--dry-run] \
  [--only-missing] \
  [--question-id ID] \
  [--approve-call-count N]
```

Builds or resumes a retrieval dataset. `--dry-run` plans without API calls.
`--only-missing` resumes only missing artifacts. `--approve-call-count N` is
required for live execution and must exactly match the plan.

Placeholder KB IDs are rejected before any API call.

### report

```bash
rag-evals report DEFINITION
```

Joins lineage across datasets and generates `reports/latest/report.csv` and
`reports/latest/report.md`. Makes zero API calls.

### status

```bash
rag-evals status TYPE DATASET_ID
```

Displays lifecycle (`building`/`complete`/`sealed`/`failed`), planned count,
successful count, failed count, and missing count.

### seal

```bash
rag-evals seal TYPE DATASET_ID
```

Transitions a complete dataset to sealed. Rejects if not complete or already
sealed.

### clean-dataset

```bash
rag-evals clean-dataset TYPE DATASET_ID [--confirm-dataset-id ID]
```

Without `--confirm-dataset-id`: displays a dry-run inventory (lifecycle, file
count, size, downstream references, report references, can-clean status).

With `--confirm-dataset-id`: atomically moves the dataset to
`datasets/.trash/<type>/<id>-<timestamp>`. Rejects sealed datasets,
downstream-referenced datasets, and mismatched confirmation IDs.

## Exit Codes

| Code | Meaning | Error Class |
|---:|---|---|
| 0 | Success | - |
| 1 | General error | `RagEvalsError` |
| 2 | CLI usage error | Typer |
| 3 | Configuration error | `ConfigError` |
| 4 | Planning error | `PlanError` |
| 5 | Artifact store error | `ArtifactError` |
| 6 | API adapter error | `AdapterError` |
| 7 | Execution error | `ExecutionError` |

Configuration, artifact, adapter, and execution failures retain their documented
domain exit codes.

## Call Approval

Any live execution with planned API calls requires the exact current count:

```bash
rag-evals run-retrieval path/to/test.yaml --approve-call-count 30
```

If filtering or source changes alter the plan, the command fails and prints the
new count. `--yes` is intentionally not provided.

## Cleanup Safety

The initial invocation is a dry run:

```bash
rag-evals clean-dataset retrieval my-dataset
```

Execution requires exact confirmation:

```bash
rag-evals clean-dataset retrieval my-dataset \
  --confirm-dataset-id my-dataset
```

The command refuses globs, prefixes, sealed datasets, and datasets referenced by
downstream manifests. It moves data to `.trash`; it does not permanently delete
it.
