# Latest Undocumented Changes

Generated: 2026-07-14

## Scope

This is a light gap report for recent work that is not yet cataloged in the
main README, file catalogs, tutorials, or research workflow documents. It is
an engineering snapshot, not predictive-quality evidence.

- Branch: `codex/rawseq-trade-flow-evolution-integration-20260714`
- HEAD: `7bddf3d`
- Safety: CPU/paper research only; no private API, orders, promotion, champion
  mutation, dashboard mutation, or active future-shadow mutation.

## Newly Committed Work

### Binance trade normalization and source audits

Commits `c3a892e` through `f5bdf6f` added canonical Binance raw-trade and
aggregate-trade normalization, aggressor-side provenance, streaming one-minute
aggregation, trade-order helpers, SOL source/coverage audits, and an inert
coverage-audit runbook.

The source contract fails closed when maker-side semantics, hashes, coverage,
or development cutoffs cannot be verified. Raw trades and overlapping
`aggTrades` cannot be combined in one contract.

### Trade-flow feature-family evolution integration

Commits `f1470be`, `86157b9`, and `6f9fbb9` added:

- hashed source contracts and strict pooled/subset/single-symbol policies;
- atomic, content-addressed minute-feature caches;
- core flow, size-distribution, timing/burst, persistence, and price-impact
  feature families;
- exact candidate filtering before trade loading;
- formula-sensitive candidate/cache/checkpoint lineage and matched
  parent-child comparisons;
- source and second-stage data preflights before model fitting.

Primary files:

- `scripts/tiny/rawseq_1m_trade_flow_evolution.py`
- `scripts/tiny/rawseq_1m_trade_flow_features.py`
- `scripts/tiny/run_rawseq_1m_feature_family_evolution.py`
- `scripts/tiny/run_rawseq_1m_board_member_target_feature_tournament.py`

### Trade-resolved barrier labels

Commit `2756b49` added a distinct trade-resolved barrier lane. When both candle
barriers occur in one minute, ordered trade timestamps and trade IDs determine
the first crossing when possible. Unresolvable rows remain explicitly marked;
the candle-only and trade-resolved lanes are not silently mixed.

### Tests, handoff, and integration packet

Commit `7bddf3d` added focused cache/coverage/identity/barrier tests, an inert
PowerShell runbook, and an integration packet builder.

- Runbook: `docs/rawseq_1m_trade_flow_evolution_manual_runbook.ps1`
- Packet builder:
  `scripts/tiny/report_rawseq_1m_trade_flow_evolution_integration.py`
- Latest packet:
  `F:\rsio\rawseq_1m_trade_flow_evolution_integration\rawseq_1m_trade_flow_evolution_integration_20260715T001721Z`
- Focused verification: 152 tests passed.
- Formula registry SHA256:
  `e7fb1f5ac8991a6031fdd13fef35217649f79e29ad7d6e10a2d307bcf99f4273`.

The real-source preflight returned
`TRADE_FLOW_EVOLUTION_BLOCKED_INSUFFICIENT_COVERAGE` before training. This is
the expected data-coverage block, not a model failure. No substantial
trade-flow tournament was launched.

### Aborted giant benchmark classification

The earlier giant feature-evolution process stopped while writing cache data
because the drive reported `No space left on device`. It remains classified as
an infrastructure/storage interruption, not model-quality evidence. Its cache
or checkpoint state should be audited before any resume attempt.

## Uncommitted Work

Separate working-tree changes remain under review and are not part of the
trade-flow integration lineage:

- live-paper dashboard history, source identity, freshness, and fail-closed
  gap handling;
- console prediction source reporting;
- board triage summary/status output;
- calibrated downside/upside confirmation artifact retention and parity
  reporting;
- methodology-supersession reporting for older calibration and expansion
  artifacts;
- related tests and npm aliases.

New untracked methodology files:

- `scripts/tiny/report_rawseq_1m_methodology_supersession.py`
- `tests/test_rawseq_1m_methodology_supersession.py`

These changes should be reviewed and committed separately before being
described as repository-stable.

## Missing From Main Documentation

None of the new trade-flow integration files are currently listed in:

- `MEGA_README.md`
- `FILE_CATALOG.md`
- `FILE_CATALOG.csv`
- `docs/RECENT_CHANGES_REPORT.md`
- `docs/RESEARCH_WORKFLOWS.md`
- `docs/AI_PROJECT_TUTORIALS.txt`

Recommended documentation follow-up:

1. Add the source contract, cache, feature families, preflight states, and
   runbook commands to the research workflow and tutorial documents.
2. Regenerate both file catalogs after the integration branch is finalized.
3. Document candle-only versus trade-resolved barrier semantics explicitly.
4. Document the dashboard changes only after their separate review/commit.

## Current Interpretation

The trade-flow integration is engineering-ready to fail closed, but the local
verified market-trade history does not cover the required pooled development
contract. It must remain blocked before fitting until source coverage passes.
No candidate from this work is credible, clean, freezeable, or promotable.
