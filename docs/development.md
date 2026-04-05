# Development

This page covers local setup for contributors.

## :material-source-branch: Clone the repository

```bash
git clone https://github.com/harens/AnomaLog.git
cd AnomaLog
```

## :material-puzzle: Install development dependencies

AnomaLog uses `uv` for local development.

```bash
uv sync --all-groups
```

That installs the main package plus the `dev` and `docs` dependency groups.

## :material-tools: Install pre-commit hooks

```bash
uv run pre-commit install
```

## :material-check-circle-outline: Run the required checks

Before opening a change, run the same validation commands used for the project:

```bash
uv run ruff format
uv run ruff check --fix
uv run ty check
uv run pytest --doctest-modules --cov=anomalog --cov-context=test --cov-report term-missing tests
```

## :material-book-open-page-variant-outline: Build the documentation locally

```bash
uv run --group docs zensical build
```

If you want to iterate on the docs locally, use the equivalent serve command supported by your docs setup.

## :material-compass-outline: Where to look in the codebase

- `anomalog/` contains the library itself
- `anomalog/_runtime/` contains internal orchestration code
- `experiments/` contains the experiment runner layer
- `docs/` contains the documentation site
- `tests/` contains unit and integration tests

For the module map, see [API Reference](reference/index.md).

## :material-database-settings-outline: How structured storage works

The default sink is `ParquetStructuredSink`.

At a high level:

- raw lines are parsed into structured records
- records are written as a partitioned Parquet dataset
- partitioning is based on a stable entity-hash bucket
- entity-based grouping reads bucket partitions and then groups rows by entity
- time-based grouping re-merges rows from bucket partitions into global timestamp order with a heap

In plain terms:

- grouping by entity is efficient because related entities land in deterministic bucket partitions
- grouping by time needs an extra merge step because time order spans multiple buckets

This is why the code in `anomalog/parsers/structured/parquet/` matters if you are changing grouping behavior or storage layout.

## :material-cached: How caching works

Caching is handled through the helpers in `anomalog/cache/` and the internal runtime in `anomalog/_runtime/`.

The important design points are:

- dataset source materialisation is tied to the dataset root under `data_root`
- derived artifacts live under `cache_root`
- structured data writes are materialised against the raw log asset path
- template training is materialised against the trained parser output path
- local output existence is checked defensively after Prefect returns, because a cached completed state alone is not enough to guarantee the artifact still exists on disk

In practice, that means:

- if you keep the same raw logs and parser, the structured stage can be reused
- if you keep the same structured data and template setup, template mining can be reused
- if an expected local artifact has been deleted, the helper will rerun the work rather than trusting the cache state blindly

If you are debugging stale outputs or changing cache behavior, start with:

- `anomalog/cache/__init__.py`
- `anomalog/_runtime/services.py`
- `anomalog/parsers/structured/parquet/sink.py`
- `anomalog/parsers/template/parsers.py`
