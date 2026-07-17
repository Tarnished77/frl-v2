# Data dictionary

## Formal experiment folders

Each folder under `formal/` contains:

- `agent_survival.csv`: one terminal record per investor-condition run.
- `run_daily.csv`: one aggregate record per run-day.
- `run_summary.csv`: one aggregate record per run.
- `seed_manifest.json`: deterministic stream identifiers and derived seeds.
- `experiment_manifest.json`: design, counts, file hashes, and provenance.

The sample folder uses the same schema.

## Shared identifiers

| Field | Definition |
| --- | --- |
| `run_id` | Unique condition and replication combination |
| `replication_id` | Paired random-stream block identifier |
| `condition_id` | Encodes calibration ratio and stress intensities |
| `population_design` | `heterogeneous` or `homogeneous` |
| `calibration_ratio` | Baseline liquidity-to-market expected-loss ratio |
| `liquidity_intensity` | Pre-specified liquidity stress level: 0, 1, or 2 |
| `market_intensity` | Pre-specified market stress level: 0, 1, or 2 |

## `agent_survival.csv`

| Field group | Definition |
| --- | --- |
| `agent_id` | Investor index within a run |
| `event` | 1 for exit by day 30; 0 for administrative censoring |
| `exit_day` | Exit day; 0 for censored observations |
| `survival_time` | Observed days until exit or censoring |
| `exit_reason` | `liquidity`, `insolvency`, `joint`, or `censored` |
| `initial_*` | Initial normalized balance-sheet quantities |
| `final_*` | Exit-time or day-30 balance-sheet quantities |
| `total_payment` | Cumulative mandatory liquidity payment |
| `total_unpaid` | Cumulative unmet payment |
| `total_forced_sale_proceeds` | Cumulative proceeds from forced sales |

## `run_daily.csv`

| Field group | Definition |
| --- | --- |
| `day` | Simulation day |
| `at_risk_start`, `at_risk_end` | Active investors before and after update |
| `*_exit_count` | Total and reason-specific daily exits |
| `market_event` | Indicator for the common adverse market event |
| `market_loss_fraction` | Direct risky-asset loss fraction for the day |
| `price_impact_loss` | Endogenous loss fraction from aggregate forced sales |
| `asset_price` | End-of-day risky-asset price index |
| `preimpact_forced_sale_value` | Requested risky-asset liquidation value |
| `forced_sale_proceeds` | Realized aggregate sale proceeds |
| `total_obligation`, `total_unpaid` | Aggregate payment need and shortfall |
| `mean_active_equity`, `mean_active_cash` | Active-investor end-of-day means |

## `run_summary.csv`

| Field group | Definition |
| --- | --- |
| `population`, `horizon` | Investors per run and maximum follow-up days |
| `stress_budget` | Fixed total reference expected-loss budget |
| `liquidity_budget`, `market_budget` | Budget allocation by channel |
| `market_event_loss_fraction` | Calibrated loss conditional on an event |
| `event_count`, `exit_rate` | Run-level exits and day-30 exit proportion |
| `rmst_days` | Run-level 30-day restricted mean survival time |
| `final_asset_price` | Day-30 price index |
| `mean_final_*` | Run-level means of terminal balance-sheet quantities |

## Compact results

`results/` contains analysis-ready summaries, paired block-level contrasts,
bootstrap intervals, time-pattern diagnostics, sensitivity results, and the
machine-readable `results_manifest.json`. `results/input_manifests/` preserves
the formal input hashes without duplicating the large raw CSV files.
