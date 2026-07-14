# Changelog

All notable changes to this project are documented in this file.

## [0.3.0] 

### Removed

- **Breaking:** the deprecated `fold_change` output column has been removed. It was an
  alias for `log2_fold_change` (identical values); use `log2_fold_change` directly. The
  `FutureWarning` previously emitted on every `pdex(...)` call is also gone.

### Changed

- `"all"` mode (1-vs-rest) is now a genuine one-shot computation instead of a per-group
  loop: the expression matrix is materialized once, each gene is ranked once via
  `mwu_one_vs_rest()`, and each group's "rest" pseudobulk/CPM is derived algebraically
  from the global sum rather than by re-slicing and re-ranking a fresh "rest" matrix.
  This turns `_pdex_all` from `O(n_groups × n_obs)` into ~`O(n_obs)`, which matters most
  for screens with many groups (e.g. guides) and/or large cell counts.
- Bumped the `numba-mwu` dependency floor to `>=0.2.0` (required by the `"all"`-mode
  optimization above).

### Fixed

- Guarded against floating-point cancellation noise in the algebraic "rest" pseudobulk
  and CPM derivation (`_pdex_all`): the rest mean of non-negative data can never be
  legitimately negative, so it is now clipped to `>= 0` before feeding
  `log2_fold_change`/`percent_change`, preventing spurious negative values from
  floating-point noise.
- `_available_cpus()` now checks `hasattr(os, "sched_getaffinity")` before calling it,
  instead of relying on `AttributeError` from the call itself, fixing a case where the
  attribute exists but raises for an unrelated reason on some platforms.

### Tests

- Added regression coverage for `"all"` mode with multiple groups and edge cases
  (`tests/test_math.py`, `tests/test_pdex.py`, `tests/conftest.py`).

### CI

- `semver-check` now only runs on pull requests targeting `main` (previously it could
  run — and fail spuriously — on unrelated PR bases).
- CI workflows now avoid redundant duplicate runs on a single PR push.

## [0.2.5] - previous release

See git history prior to this file's introduction for earlier changes.
