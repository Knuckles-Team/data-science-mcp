---
name: data-science-quant-analytics
skill_type: skill
description: >-
  Run the medulla financial-quant kernels through the data-science-mcp MCP
  server — market-making quotes, order-flow microstructure, position sizing,
  backtest validation, state-space filters, signal combination, derivatives
  surfaces and forensic accounting. Use when the agent must quote, size, validate
  or filter a trading strategy from numeric series. Do NOT use for general
  model fitting (use data-science-model-training) or to place orders.
license: MIT
tags: [data-science, quant, finance, mcp]
metadata:
  author: Knuckles-Team
  version: '0.1.0'
---
# Data Science Quant Analytics

Every kernel is its own typed tool, `quant_<domain>_<kernel>`. All compute runs in
the medulla engine; with no engine reachable each tool returns an `error` payload.
List arguments are JSON strings (`*_json`).

## Tools
| Domain | Tools |
|--------|-------|
| Market making | `quant_market_making_avellaneda_stoikov`, `quant_market_making_glt_quotes`, `quant_market_making_logit_quotes`, `quant_market_making_glosten_milgrom_spread`, `quant_market_making_expected_pnl_rate`, `quant_market_making_breakeven_alpha` |
| Microstructure | `quant_microstructure_ofi_series`, `quant_microstructure_microprice_series`, `quant_microstructure_vpin_pm`, `quant_microstructure_hawkes_mle`, `quant_microstructure_hardiman_bouchaud`, `quant_microstructure_queue_imbalance`, `quant_microstructure_realized_vol_tick` |
| Sizing | `quant_sizing_kelly_fraction`, `quant_sizing_empirical_kelly`, `quant_sizing_bayesian_kelly`, `quant_sizing_posterior_credible_interval` |
| Validation | `quant_validation_purged_cpcv`, `quant_validation_deflated_sharpe`, `quant_validation_probability_backtest_overfit`, `quant_validation_diebold_mariano`, `quant_validation_brier_score` |
| State space | `quant_statespace_kalman_filter`, `quant_statespace_kalman_beta`, `quant_statespace_kalman_volatility`, `quant_statespace_adf_test`, `quant_statespace_ou_calibrate`, `quant_statespace_ou_optimal_thresholds`, `quant_statespace_markov_transition` |
| Signals | `quant_signals_order_book_imbalance`, `quant_signals_information_ratio`, `quant_signals_effective_independent_n`, `quant_signals_alpha_combination`, `quant_signals_convergence_gate`, `quant_signals_spread_reversion` |
| Derivatives | `quant_derivatives_implied_vol`, `quant_derivatives_smile`, `quant_derivatives_calibrate` |
| Forensic | `quant_forensic` |

## Steps
1. Estimate the inputs: `quant_microstructure_*` and `quant_statespace_*` on the raw series.
2. Combine signals with `quant_signals_*` and size with `quant_sizing_*`.
3. Quote with `quant_market_making_*` using the sized inventory and volatility.
4. Validate before trusting: `quant_validation_purged_cpcv`, then `quant_validation_deflated_sharpe` and `quant_validation_probability_backtest_overfit`.

## Failure modes
- Malformed `*_json` arguments return a clean error naming the argument.
- An engine or client one version behind reports an unknown-kernel error; upgrade it.
