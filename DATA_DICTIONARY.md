# Data dictionary

## Formal block-level data

Each `data/formal/**/run_summary.csv` row is one replication block under one
stress condition.

| Field | Meaning |
|---|---|
| `run_id` | Unique replication-condition identifier. |
| `replication_id` | Paired Monte Carlo block identifier. |
| `condition_id` | Locked stress-cell identifier. |
| `calibration_ratio` | Funding-to-market allocation ratio, rho. |
| `funding_intensity` | Funding-withdrawal intensity: 0, 1, or 2. |
| `market_intensity` | Market-loss intensity: 0, 1, or 2. |
| `price_impact_lambda` | Exponential aggregate-sale price-impact parameter. |
| `market_event_probability` | Daily common market-event probability. |
| `cash_scenario` | Baseline, low-cash, or high-cash setting. |
| `population` | Institutions in the replication block. |
| `horizon` | Simulation days. |
| `stress_budget` | Thirty-day normalized reference stress budget. |
| `funding_budget` | Reference funding allocation before intensity scaling. |
| `market_budget` | Reference market allocation before intensity scaling. |
| `market_event_loss_fraction` | Loss conditional on a market event. |
| `target_cumulative_funding` | Reference cumulative funding withdrawal. |
| `target_cumulative_market_loss` | Reference cumulative market-value loss. |
| `event_count` | Common market events realized in the block. |
| `exit_rate` | Fraction exiting by day 30. |
| `rmst_days` | Run-level restricted mean survival time; retained as a secondary simulation diagnostic only. |
| `final_asset_price` | Day-30 common risky-asset price. |
| `asset_price_loss` | Loss relative to the initial common asset price. |
| `mean_final_equity` | Mean day-30 institutional equity. |
| `mean_equity_loss` | Mean loss relative to normalized initial equity. |
| `p90_equity_loss` | Ninetieth percentile of institutional equity loss. |
| `worst_10pct_mean_equity_loss` | Mean loss among the worst-loss institutional decile. |
| `mean_final_cash` | Mean day-30 cash. |
| `mean_final_liabilities` | Mean day-30 short-term debt. |
| `mean_liability_reduction` | Mean reduction in debt from its initial level. |
| `mean_total_funding_requested` | Mean cumulative funding requested. |
| `mean_total_funding_repaid` | Mean cumulative funding repaid. |
| `mean_total_unpaid` | Mean cumulative unpaid funding. |
| `mean_forced_sale_proceeds` | Mean cumulative forced-sale proceeds. |
| `mean_market_value_loss` | Mean direct market-repricing component of equity loss. |
| `mean_fire_sale_value_loss` | Mean endogenous price-impact component of equity loss. |

The compact `experiment_manifest.json` lists only redistributed files.
`experiment_manifest.full.json` preserves the original manifest and hashes for
the omitted agent-level and daily files.

## Derived primary cells

`derived-resilience-primary-v1/resilience_cells.csv` contains one row per
outcome and stress cell. `estimate` is the block mean; `block_se` is the Monte
Carlo standard error. Pointwise and family-wise simultaneous interval bounds
are reported separately.

`resilience_contrasts.csv` contains local channel effects, fire-sale add-ons,
and channel interactions. The paired replication block is the resampling unit.

## Sensitivity data

Each `derived-resilience-sensitivity-*/sensitivity_cells.csv` reports scenario
cell means. Each `sensitivity_contrasts.csv` reports scenario-minus-reference
paired contrasts. Market-event-frequency scenarios preserve the expected
30-day market-loss target.

## Publication data

`expected/publication/data/` contains the compact CSV payload used for final
tables and figures. Its complete file hashes are recorded in
`expected/publication/publication_manifest.json`.
