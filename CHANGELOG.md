# Changelog

Not registered yet, so everything here is unreleased.

## Unreleased

- Threading moved from OhMyThreads.jl to Polyester.jl: much faster for small and
  moderate problems, same results.
- `cqk_init` vectorises its sums with `@simd`; `altmapreduce` no longer allocates
  per call.
- Removed `altforeach` (unexported, unused).
