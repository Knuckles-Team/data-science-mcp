"""MCP tools exposing the epistemic-graph Rust financial-quant kernels.

All compute runs in the Rust ``epistemic-graph`` engine over its MessagePack/UDS
client (``client.finance.*``); Python stays thin and only marshals arguments.
The engine connection is the existing cached singleton on
``MLEngine._rust_client()`` — no raw sockets are opened here.

These are **action-routed** tools: one tool per domain, dispatching on an
``action`` argument to the matching ``client.finance.*`` kernel
(CONCEPT:KG-2.20f market-making/sizing/validation, CONCEPT:KG-2.20g forensic).
"""

import json
from typing import Any

from fastmcp import Context, FastMCP
from pydantic import Field

from data_science_mcp.ml_engine import MLEngine, _ENGINE_REQUIRED_ERR


def _finance() -> Any | None:
    """Return the ``finance`` sub-client of the cached engine connection, or None
    when the engine is unreachable (callers surface a clean error)."""
    client = MLEngine._rust_client()
    if client is None:
        return None
    return client.finance


def _loads(name: str, raw: str) -> Any:
    """Parse a JSON argument, raising ``ValueError`` with a tool-friendly message."""
    try:
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001 — surfaced as an {"error": ...} dict
        raise ValueError(f"Invalid {name}: {exc}") from exc


def register_quant_tools(mcp: FastMCP) -> None:
    @mcp.tool(tags={"quant"})
    async def quant_market_making(
        action: str = Field(
            description=(
                "Quoting kernel to run: 'avellaneda_stoikov', 'glt_quotes', "
                "'logit_quotes', 'glosten_milgrom_spread', 'expected_pnl_rate', "
                "or 'breakeven_alpha'."
            )
        ),
        mid: float = Field(
            default=0.5,
            description="Mid price (avellaneda_stoikov/glt_quotes).",
        ),
        p_mid: float = Field(
            default=0.5,
            description="Bounded (0,1) mid probability (logit_quotes).",
        ),
        inventory: float = Field(default=0.0, description="Current signed inventory."),
        sigma: float = Field(default=0.0, description="Volatility estimate."),
        gamma: float = Field(default=0.1, description="Risk-aversion parameter."),
        kappa: float = Field(default=1.0, description="Order-arrival decay (κ)."),
        tau: float = Field(
            default=1.0,
            description="Time-to-horizon (avellaneda_stoikov/logit_quotes).",
        ),
        a: float = Field(
            default=1.0, description="Order-arrival intensity (glt_quotes)."
        ),
        boundary_m: float = Field(
            default=0.0,
            description="Boundary inventory cap for logit_quotes.",
        ),
        alpha: float = Field(
            default=0.0,
            description="Informed-trader fraction (glosten_milgrom/expected_pnl).",
        ),
        p: float = Field(
            default=0.5,
            description="Probability of the high value (glosten_milgrom/pnl/breakeven).",
        ),
        delta: float = Field(
            default=0.0,
            description="Half-spread offset (expected_pnl_rate/breakeven_alpha).",
        ),
        v_h: float = Field(default=1.0, description="High payoff value."),
        v_l: float = Field(default=0.0, description="Low payoff value."),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Market-making / HFT quoting kernels (CONCEPT:KG-2.20f).

        Routes ``action`` to the matching engine kernel: Avellaneda-Stoikov and
        GLFT closed-form quotes (``{bid, ask, reservation, half_spread,
        withdraw}``), logit-space quotes for bounded prediction-market prices,
        the Glosten-Milgrom adverse-selection spread, the expected PnL rate, and
        the break-even informed-trader fraction.
        """
        if ctx:
            await ctx.info(f"quant_market_making action={action}")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            if action == "avellaneda_stoikov":
                return fin.avellaneda_stoikov(mid, inventory, sigma, gamma, kappa, tau)
            if action == "glt_quotes":
                return fin.glt_quotes(mid, inventory, sigma, gamma, kappa, a)
            if action == "logit_quotes":
                return fin.logit_quotes(
                    p_mid, inventory, sigma, gamma, kappa, tau, boundary_m
                )
            if action == "glosten_milgrom_spread":
                return {"spread": fin.glosten_milgrom_spread(alpha, p)}
            if action == "expected_pnl_rate":
                return {
                    "pnl_rate": fin.expected_pnl_rate(
                        delta, a, kappa, alpha, p, v_h, v_l
                    )
                }
            if action == "breakeven_alpha":
                return {"breakeven_alpha": fin.breakeven_alpha(delta, p, v_h, v_l)}
            return {"error": f"Unknown action: {action}"}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @mcp.tool(tags={"quant"})
    async def quant_microstructure(
        action: str = Field(
            description=(
                "Microstructure kernel: 'ofi_series', 'microprice_series', "
                "'vpin_pm', 'hawkes_mle', 'hardiman_bouchaud', "
                "'queue_imbalance', or 'realized_vol_tick'."
            )
        ),
        ts_json: str = Field(
            default="[]", description="JSON list of event timestamps (ofi_series)."
        ),
        bid_px_json: str = Field(
            default="[]", description="JSON list of bid prices (ofi/microprice)."
        ),
        bid_sz_json: str = Field(
            default="[]", description="JSON list of bid sizes (ofi/microprice)."
        ),
        ask_px_json: str = Field(
            default="[]", description="JSON list of ask prices (ofi/microprice)."
        ),
        ask_sz_json: str = Field(
            default="[]", description="JSON list of ask sizes (ofi/microprice)."
        ),
        window_secs: float = Field(
            default=1.0, description="Rolling window in seconds (ofi_series)."
        ),
        buy_vol_json: str = Field(
            default="[]", description="JSON list of bucket buy volumes (vpin_pm)."
        ),
        sell_vol_json: str = Field(
            default="[]", description="JSON list of bucket sell volumes (vpin_pm)."
        ),
        p_mean_json: str = Field(
            default="[]", description="JSON list of bucket mean prices (vpin_pm)."
        ),
        times_json: str = Field(
            default="[]",
            description="JSON list of event times (hawkes_mle/hardiman_bouchaud).",
        ),
        t_horizon: float = Field(
            default=1.0,
            description="Observation horizon (hawkes_mle/hardiman_bouchaud).",
        ),
        max_iter: int = Field(
            default=200, description="Max MLE iterations (hawkes_mle)."
        ),
        n_windows: int = Field(
            default=100, description="Count windows (hardiman_bouchaud)."
        ),
        bid_q_json: str = Field(
            default="[]", description="JSON list of best-bid queue sizes (queue_imbalance)."
        ),
        ask_q_json: str = Field(
            default="[]", description="JSON list of best-ask queue sizes (queue_imbalance)."
        ),
        bid_rate_json: str = Field(
            default="[]",
            description="JSON list of bid-side arrival rates (queue_imbalance; uniform if empty).",
        ),
        ask_rate_json: str = Field(
            default="[]",
            description="JSON list of ask-side arrival rates (queue_imbalance; uniform if empty).",
        ),
        mid_json: str = Field(
            default="[]", description="JSON list of mid-prices (realized_vol_tick)."
        ),
        window: int = Field(
            default=20,
            description="Rolling tick window (queue arrival default / realized_vol_tick).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Order-flow / toxicity / self-excitation kernels (CONCEPT:KG-2.20f).

        Routes ``action`` to: Cont-Kukanov-Stoikov order-flow imbalance series,
        size-weighted microprice series, prediction-market VPIN toxicity, a
        Hawkes-process MLE fit (returns branching ratio / half-life), and the
        model-free Hardiman-Bouchaud branching-ratio estimate.
        """
        if ctx:
            await ctx.info(f"quant_microstructure action={action}")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            if action == "ofi_series":
                return {
                    "ofi": fin.ofi_series(
                        _loads("ts_json", ts_json),
                        _loads("bid_px_json", bid_px_json),
                        _loads("bid_sz_json", bid_sz_json),
                        _loads("ask_px_json", ask_px_json),
                        _loads("ask_sz_json", ask_sz_json),
                        window_secs,
                    )
                }
            if action == "microprice_series":
                return {
                    "microprice": fin.microprice_series(
                        _loads("bid_px_json", bid_px_json),
                        _loads("bid_sz_json", bid_sz_json),
                        _loads("ask_px_json", ask_px_json),
                        _loads("ask_sz_json", ask_sz_json),
                    )
                }
            if action == "vpin_pm":
                return {
                    "vpin": fin.vpin_pm(
                        _loads("buy_vol_json", buy_vol_json),
                        _loads("sell_vol_json", sell_vol_json),
                        _loads("p_mean_json", p_mean_json),
                    )
                }
            if action == "hawkes_mle":
                return fin.hawkes_mle(
                    _loads("times_json", times_json), t_horizon, max_iter
                )
            if action == "hardiman_bouchaud":
                return {
                    "branching_ratio": fin.hardiman_bouchaud(
                        _loads("times_json", times_json), t_horizon, n_windows
                    )
                }
            if action == "queue_imbalance":
                return fin.queue_imbalance(
                    _loads("bid_q_json", bid_q_json),
                    _loads("ask_q_json", ask_q_json),
                    _loads("bid_rate_json", bid_rate_json),
                    _loads("ask_rate_json", ask_rate_json),
                )
            if action == "realized_vol_tick":
                return {
                    "realized_vol": fin.realized_vol_tick(
                        _loads("mid_json", mid_json), window
                    )
                }
            return {"error": f"Unknown action: {action}"}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @mcp.tool(tags={"quant"})
    async def quant_sizing(
        action: str = Field(
            description=(
                "Sizing kernel: 'kelly_fraction', 'bayesian_kelly', "
                "'posterior_credible_interval', or 'empirical_kelly'."
            )
        ),
        q: float = Field(
            default=0.5, description="Estimated win probability (kelly_fraction)."
        ),
        c: float = Field(
            default=0.5,
            description="Contract cost / price (kelly_fraction/bayesian_kelly).",
        ),
        fraction: float = Field(
            default=0.25, description="Kelly scaling fraction (kelly_fraction)."
        ),
        alpha: float = Field(
            default=1.0,
            description="Beta posterior α (bayesian_kelly/credible_interval).",
        ),
        beta: float = Field(
            default=1.0,
            description="Beta posterior β (bayesian_kelly/credible_interval).",
        ),
        n_quadrature: int = Field(
            default=50, description="Quadrature nodes (bayesian_kelly)."
        ),
        level: float = Field(
            default=0.05,
            description="Tail level for the credible interval (e.g. 0.05 → 95%).",
        ),
        p: float = Field(
            default=0.5, description="Win probability p (empirical_kelly)."
        ),
        b: float = Field(default=1.0, description="Net win odds b (empirical_kelly)."),
        historical_returns_json: str = Field(
            default="[]",
            description="JSON list of historical per-bet returns (empirical_kelly).",
        ),
        n_simulations: int = Field(
            default=10000,
            description="Monte-Carlo resamples for the empirical Kelly (empirical_kelly).",
        ),
        seed: int = Field(
            default=42, description="RNG seed for empirical Kelly resampling."
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Position-sizing kernels (CONCEPT:KG-2.20f / KG-2.20i).

        Routes ``action`` to fractional Kelly for a YES contract, Bayesian Kelly
        under a Beta(α,β) posterior (shrinks the bet as variance grows), the
        posterior credible interval ``{lower, upper}``, and the empirical
        (Monte-Carlo resampled) Kelly fraction that sizes against the realized
        return distribution rather than a single point edge.
        """
        if ctx:
            await ctx.info(f"quant_sizing action={action}")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            if action == "kelly_fraction":
                return {"fraction": fin.kelly_fraction(q, c, fraction)}
            if action == "bayesian_kelly":
                return {"fraction": fin.bayesian_kelly(alpha, beta, c, n_quadrature)}
            if action == "posterior_credible_interval":
                return fin.posterior_credible_interval(alpha, beta, level)
            if action == "empirical_kelly":
                return {
                    "fraction": fin.empirical_kelly(
                        p,
                        b,
                        _loads("historical_returns_json", historical_returns_json),
                        n_simulations,
                        seed,
                    )
                }
            return {"error": f"Unknown action: {action}"}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @mcp.tool(tags={"quant"})
    async def quant_validation(
        action: str = Field(
            description=(
                "Validation kernel: 'purged_cpcv', 'deflated_sharpe', "
                "'probability_backtest_overfit', 'diebold_mariano', or "
                "'brier_score'."
            )
        ),
        n_samples: int = Field(
            default=0, description="Number of samples (purged_cpcv)."
        ),
        n_groups: int = Field(default=6, description="CV groups (purged_cpcv)."),
        n_test_groups: int = Field(
            default=2, description="Test groups per split (purged_cpcv)."
        ),
        purge_window: int = Field(
            default=0, description="Purge window in samples (purged_cpcv)."
        ),
        embargo: int = Field(
            default=0, description="Embargo in samples (purged_cpcv)."
        ),
        observed_sr: float = Field(
            default=0.0, description="Observed Sharpe ratio (deflated_sharpe)."
        ),
        n_trials: int = Field(
            default=1, description="Number of strategy trials (deflated_sharpe)."
        ),
        sr_returns_json: str = Field(
            default="[]",
            description="JSON list of strategy returns (deflated_sharpe).",
        ),
        insample_json: str = Field(
            default="[]",
            description="JSON 2D matrix rows=splits, cols=strategies (PBO).",
        ),
        oos_json: str = Field(
            default="[]",
            description="JSON 2D matrix of out-of-sample performances (PBO).",
        ),
        losses_a_json: str = Field(
            default="[]", description="JSON list of model-A losses (diebold_mariano)."
        ),
        losses_b_json: str = Field(
            default="[]", description="JSON list of model-B losses (diebold_mariano)."
        ),
        h: int = Field(
            default=1, description="Forecast horizon for HAC (diebold_mariano)."
        ),
        forecasts_json: str = Field(
            default="[]",
            description="JSON list of probability forecasts in [0,1] (brier_score).",
        ),
        outcomes_json: str = Field(
            default="[]",
            description="JSON list of realized 0/1 outcomes (brier_score).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Backtest-validation / calibration kernels (CONCEPT:KG-2.20f / KG-2.20i).

        Routes ``action`` to purged combinatorial CV splits, the deflated Sharpe
        ratio, the probability of backtest overfitting, the Diebold-Mariano
        equal-predictive-accuracy test ``{statistic, p_value, a_better}``, and the
        Brier score of probabilistic forecasts vs realized outcomes (lower = better
        calibrated).
        """
        if ctx:
            await ctx.info(f"quant_validation action={action}")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            if action == "purged_cpcv":
                return {
                    "splits": fin.purged_cpcv(
                        n_samples, n_groups, n_test_groups, purge_window, embargo
                    )
                }
            if action == "deflated_sharpe":
                return {
                    "deflated_sharpe": fin.deflated_sharpe(
                        observed_sr,
                        n_trials,
                        _loads("sr_returns_json", sr_returns_json),
                    )
                }
            if action == "probability_backtest_overfit":
                return {
                    "pbo": fin.probability_backtest_overfit(
                        _loads("insample_json", insample_json),
                        _loads("oos_json", oos_json),
                    )
                }
            if action == "diebold_mariano":
                return fin.diebold_mariano(
                    _loads("losses_a_json", losses_a_json),
                    _loads("losses_b_json", losses_b_json),
                    h,
                )
            if action == "brier_score":
                return {
                    "brier_score": fin.brier_score(
                        _loads("forecasts_json", forecasts_json),
                        _loads("outcomes_json", outcomes_json),
                    )
                }
            return {"error": f"Unknown action: {action}"}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @mcp.tool(tags={"quant"})
    async def quant_statespace(
        action: str = Field(
            description=(
                "State-space / stat-arb kernel: 'kalman_filter', 'kalman_beta', "
                "'kalman_volatility', 'adf_test', 'ou_calibrate', "
                "'ou_optimal_thresholds', or 'markov_transition'."
            )
        ),
        observations_json: str = Field(
            default="[]",
            description="JSON list of scalar observations (kalman_filter).",
        ),
        f: float = Field(
            default=1.0, description="State-transition factor (kalman_filter)."
        ),
        q: float = Field(
            default=1e-5,
            description="Process-noise variance (kalman_filter/kalman_beta).",
        ),
        h: float = Field(
            default=1.0, description="Observation factor (kalman_filter)."
        ),
        r: float = Field(
            default=1e-3,
            description="Observation-noise variance (kalman_filter/kalman_beta).",
        ),
        x0: float = Field(default=0.0, description="Initial state (kalman_filter)."),
        p0: float = Field(
            default=1.0,
            description="Initial state variance (kalman_filter/beta/volatility).",
        ),
        market_returns_json: str = Field(
            default="[]",
            description="JSON list of market returns (kalman_beta).",
        ),
        asset_returns_json: str = Field(
            default="[]",
            description="JSON list of asset returns (kalman_beta).",
        ),
        beta0: float = Field(default=1.0, description="Initial beta (kalman_beta)."),
        returns_json: str = Field(
            default="[]",
            description="JSON list of returns (kalman_volatility).",
        ),
        q_vol: float = Field(
            default=0.1, description="Log-variance process noise (kalman_volatility)."
        ),
        r_vol: float = Field(
            default=1.0, description="Log-variance obs noise (kalman_volatility)."
        ),
        log_var0: float | None = Field(
            default=None, description="Initial log-variance (kalman_volatility)."
        ),
        annualization: float = Field(
            default=252.0, description="Annualization factor (kalman_volatility)."
        ),
        series_json: str = Field(
            default="[]",
            description="JSON list — the test series (adf_test).",
        ),
        max_lag: int = Field(default=1, description="Max ADF lag (adf_test)."),
        spread_json: str = Field(
            default="[]",
            description="JSON list — the mean-reverting spread (ou_calibrate).",
        ),
        dt: float = Field(default=1.0, description="Sampling interval (ou_calibrate)."),
        theta: float = Field(
            default=0.0, description="OU mean-reversion speed (ou_optimal_thresholds)."
        ),
        mu: float = Field(
            default=0.0, description="OU long-run mean (ou_optimal_thresholds)."
        ),
        sigma: float = Field(
            default=0.0, description="OU diffusion (ou_optimal_thresholds)."
        ),
        sigma_eq: float = Field(
            default=0.0,
            description="OU equilibrium std-dev (ou_optimal_thresholds).",
        ),
        cost: float = Field(
            default=0.0, description="Round-trip cost (ou_optimal_thresholds)."
        ),
        states_json: str = Field(
            default="[]",
            description="JSON list of integer states (markov_transition).",
        ),
        n_states: int = Field(
            default=2, description="Number of discrete states (markov_transition)."
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """State-space / statistical-arbitrage kernels (CONCEPT:KG-2.20h).

        Routes ``action`` to a 1-D Kalman filter ``{states, variances}``, a
        time-varying CAPM-beta Kalman filter ``{states, variances}``, a Kalman
        stochastic-volatility estimator (annualized vol series), the Augmented
        Dickey-Fuller stationarity test, Ornstein-Uhlenbeck calibration
        ``{theta, mu, sigma, half_life, sigma_eq}``, the OU optimal entry/exit
        thresholds, and an empirical Markov transition matrix.
        """
        if ctx:
            await ctx.info(f"quant_statespace action={action}")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            if action == "kalman_filter":
                return fin.kalman_filter_1d(
                    _loads("observations_json", observations_json),
                    f,
                    q,
                    h,
                    r,
                    x0,
                    p0,
                )
            if action == "kalman_beta":
                return fin.kalman_beta(
                    _loads("market_returns_json", market_returns_json),
                    _loads("asset_returns_json", asset_returns_json),
                    q,
                    r,
                    beta0,
                    p0,
                )
            if action == "kalman_volatility":
                return {
                    "annualized_vol": fin.kalman_volatility(
                        _loads("returns_json", returns_json),
                        q_vol,
                        r_vol,
                        log_var0,
                        p0,
                        annualization,
                    )
                }
            if action == "adf_test":
                return fin.adf_test(_loads("series_json", series_json), max_lag)
            if action == "ou_calibrate":
                return fin.ou_calibrate(_loads("spread_json", spread_json), dt)
            if action == "ou_optimal_thresholds":
                return fin.ou_optimal_thresholds(theta, mu, sigma, sigma_eq, cost)
            if action == "markov_transition":
                return {
                    "transition_matrix": fin.markov_transition_matrix(
                        _loads("states_json", states_json), n_states
                    )
                }
            return {"error": f"Unknown action: {action}"}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @mcp.tool(tags={"quant"})
    async def quant_signals(
        action: str = Field(
            description=(
                "Signal-combination kernel: 'order_book_imbalance', "
                "'information_ratio', 'effective_independent_n', "
                "'alpha_combination', 'convergence_gate', or 'spread_reversion'."
            )
        ),
        v_bid_json: str = Field(
            default="[]",
            description="JSON list of bid volumes (order_book_imbalance).",
        ),
        v_ask_json: str = Field(
            default="[]",
            description="JSON list of ask volumes (order_book_imbalance).",
        ),
        bid_px_json: str = Field(
            default="[]", description="JSON list of bid prices (spread_reversion)."
        ),
        ask_px_json: str = Field(
            default="[]", description="JSON list of ask prices (spread_reversion)."
        ),
        window: int = Field(
            default=20, description="Rolling window for the spread z-score (spread_reversion)."
        ),
        ic: float = Field(
            default=0.0,
            description="Information coefficient per signal (information_ratio).",
        ),
        n_independent: float = Field(
            default=1.0,
            description="Effective independent breadth (information_ratio).",
        ),
        returns_matrix_json: str = Field(
            default="[]",
            description=(
                "JSON 2D matrix rows=signals/assets, cols=time "
                "(effective_independent_n/alpha_combination)."
            ),
        ),
        lookback: int = Field(
            default=20, description="Rolling lookback window (alpha_combination)."
        ),
        strengths_json: str = Field(
            default="[]",
            description="JSON list of per-signal strengths (convergence_gate).",
        ),
        strong_threshold: float = Field(
            default=0.6,
            description="Min |strength| to count as a strong vote (convergence_gate).",
        ),
        min_agree: int = Field(
            default=5,
            description="Min agreeing strong votes to pass the gate (convergence_gate).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Signal-combination / breadth kernels (CONCEPT:KG-2.20i).

        Routes ``action`` to order-book imbalance series, the fundamental-law
        information ratio ``IC·sqrt(N)``, the effective number of independent bets
        from a returns correlation structure, a rolling alpha-combination weight
        engine (weights sum-of-abs = 1), and a convergence gate that passes only
        when ≥``min_agree`` signals agree strongly on one direction.
        """
        if ctx:
            await ctx.info(f"quant_signals action={action}")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            if action == "order_book_imbalance":
                return {
                    "imbalance": fin.order_book_imbalance(
                        _loads("v_bid_json", v_bid_json),
                        _loads("v_ask_json", v_ask_json),
                    )
                }
            if action == "information_ratio":
                return {"information_ratio": fin.information_ratio(ic, n_independent)}
            if action == "effective_independent_n":
                return {
                    "effective_n": fin.effective_independent_n(
                        _loads("returns_matrix_json", returns_matrix_json)
                    )
                }
            if action == "alpha_combination":
                return {
                    "weights": fin.alpha_combination_engine(
                        _loads("returns_matrix_json", returns_matrix_json),
                        lookback,
                    )
                }
            if action == "convergence_gate":
                return fin.convergence_gate(
                    _loads("strengths_json", strengths_json),
                    strong_threshold,
                    min_agree,
                )
            if action == "spread_reversion":
                return fin.spread_reversion(
                    _loads("bid_px_json", bid_px_json),
                    _loads("ask_px_json", ask_px_json),
                    window,
                )
            return {"error": f"Unknown action: {action}"}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @mcp.tool(tags={"quant"})
    async def quant_forensic(
        this_year_json: str = Field(
            description=(
                "JSON dict of current fiscal-year line items (sales, cogs, sga, "
                "net_income, cfo, receivables, current_assets, current_liabilities, "
                "ppe_net, depreciation, total_assets, total_liabilities, "
                "long_term_debt, retained_earnings, ebit, market_cap, shares)."
            )
        ),
        prior_year_json: str = Field(
            description="JSON dict of prior fiscal-year line items (same keys)."
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Forensic-accounting report (CONCEPT:KG-2.20g).

        Computes Beneish M / Altman Z / Piotroski F scores and the Sloan accruals
        ratio over two fiscal years via the engine, returning
        ``{m_score, z_score, f_score, accruals_ratio, flags, verdict}``.
        """
        if ctx:
            await ctx.info("quant_forensic forensic_report")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            this_year = _loads("this_year_json", this_year_json)
            prior_year = _loads("prior_year_json", prior_year_json)
        except ValueError as exc:
            return {"error": str(exc)}
        try:
            return fin.forensic_report(this_year, prior_year)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}

    @mcp.tool(tags={"quant"})
    async def quant_derivatives(
        action: str = Field(
            description=("SABR kernel to run: 'implied_vol', 'smile', or 'calibrate'.")
        ),
        f: float = Field(default=0.0, description="Forward / underlying price."),
        k: float = Field(default=0.0, description="Strike (implied_vol)."),
        t: float = Field(default=0.0, description="Time to expiry in years."),
        alpha: float = Field(
            default=0.0, description="SABR α — initial vol (implied_vol/smile)."
        ),
        beta: float = Field(
            default=0.5,
            description="SABR β — elasticity (fixed for calibrate, default 0.5).",
        ),
        rho: float = Field(
            default=0.0,
            description="SABR ρ — spot/vol correlation (implied_vol/smile).",
        ),
        nu: float = Field(
            default=0.0, description="SABR ν — vol-of-vol (implied_vol/smile)."
        ),
        strikes_json: str = Field(
            default="[]",
            description="JSON list of strikes (smile/calibrate).",
        ),
        market_vols_json: str = Field(
            default="[]",
            description="JSON list of market implied vols to fit (calibrate).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """SABR stochastic-volatility surface kernels (CONCEPT:KG-2.20j).

        Routes ``action`` to the Hagan-2002 SABR lognormal implied vol for one
        strike, the implied-vol smile across a strike grid, and a least-squares
        SABR calibration ``{alpha, beta, rho, nu, rmse, converged}`` fitting
        (α, ρ, ν) to a market smile with β held fixed.
        """
        if ctx:
            await ctx.info(f"quant_derivatives action={action}")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            if action == "implied_vol":
                return {
                    "implied_vol": fin.sabr_implied_vol(f, k, t, alpha, beta, rho, nu)
                }
            if action == "smile":
                return {
                    "vols": fin.sabr_smile(
                        f,
                        _loads("strikes_json", strikes_json),
                        t,
                        alpha,
                        beta,
                        rho,
                        nu,
                    )
                }
            if action == "calibrate":
                return fin.sabr_calibrate(
                    f,
                    t,
                    _loads("strikes_json", strikes_json),
                    _loads("market_vols_json", market_vols_json),
                    beta,
                )
            return {"error": f"Unknown action: {action}"}
        except ValueError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
