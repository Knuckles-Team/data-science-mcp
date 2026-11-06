"""MCP tools exposing the medulla Rust financial-quant kernels.

All compute runs in the Rust ``medulla`` engine over its MessagePack/UDS
client (``client.finance.*``); Python stays thin and only marshals arguments.
The engine connection is the existing cached singleton on
``MLEngine._rust_client()`` — no raw sockets are opened here.

Each kernel is its own typed tool, ``quant_<domain>_<kernel>``, calling the matching
``client.finance.*`` kernel
(CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest market-making/sizing/validation, CONCEPT:EG-KG.domains.forensic-accounting-kernels forensic).
"""

import json
from typing import Any

from fastmcp import Context, FastMCP
from pydantic import Field

from data_science_mcp.ml_engine import _ENGINE_REQUIRED_ERR, MLEngine


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
    except Exception as exc:
        raise ValueError(f"Invalid {name}: {type(exc).__name__}") from exc


def register_quant_tools(mcp: FastMCP) -> None:
    """Register one typed tool per ``client.finance`` kernel."""

    @mcp.tool(tags={"quant"})
    async def quant_market_making_avellaneda_stoikov(
        mid: float = Field(
            default=0.5,
            description="Mid price (avellaneda_stoikov/glt_quotes).",
        ),
        inventory: float = Field(default=0.0, description="Current signed inventory."),
        sigma: float = Field(default=0.0, description="Volatility estimate."),
        gamma: float = Field(default=0.1, description="Risk-aversion parameter."),
        kappa: float = Field(default=1.0, description="Order-arrival decay (κ)."),
        tau: float = Field(
            default=1.0,
            description="Time-to-horizon (avellaneda_stoikov/logit_quotes).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Market-making / HFT quoting kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): avellaneda stoikov."""
        if ctx:
            await ctx.info("quant_market_making_avellaneda_stoikov")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.avellaneda_stoikov(mid, inventory, sigma, gamma, kappa, tau)
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_market_making_glt_quotes(
        mid: float = Field(
            default=0.5,
            description="Mid price (avellaneda_stoikov/glt_quotes).",
        ),
        inventory: float = Field(default=0.0, description="Current signed inventory."),
        sigma: float = Field(default=0.0, description="Volatility estimate."),
        gamma: float = Field(default=0.1, description="Risk-aversion parameter."),
        kappa: float = Field(default=1.0, description="Order-arrival decay (κ)."),
        a: float = Field(
            default=1.0, description="Order-arrival intensity (glt_quotes)."
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Market-making / HFT quoting kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): glt quotes."""
        if ctx:
            await ctx.info("quant_market_making_glt_quotes")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.glt_quotes(mid, inventory, sigma, gamma, kappa, a)
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_market_making_logit_quotes(
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
        boundary_m: float = Field(
            default=0.0,
            description="Boundary inventory cap for logit_quotes.",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Market-making / HFT quoting kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): logit quotes."""
        if ctx:
            await ctx.info("quant_market_making_logit_quotes")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.logit_quotes(
                p_mid, inventory, sigma, gamma, kappa, tau, boundary_m
            )
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_market_making_glosten_milgrom_spread(
        alpha: float = Field(
            default=0.0,
            description="Informed-trader fraction (glosten_milgrom/expected_pnl).",
        ),
        p: float = Field(
            default=0.5,
            description="Probability of the high value (glosten_milgrom/pnl/breakeven).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Market-making / HFT quoting kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): glosten milgrom spread."""
        if ctx:
            await ctx.info("quant_market_making_glosten_milgrom_spread")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {"spread": fin.glosten_milgrom_spread(alpha, p)}
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_market_making_expected_pnl_rate(
        kappa: float = Field(default=1.0, description="Order-arrival decay (κ)."),
        a: float = Field(
            default=1.0, description="Order-arrival intensity (glt_quotes)."
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
        """Market-making / HFT quoting kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): expected pnl rate."""
        if ctx:
            await ctx.info("quant_market_making_expected_pnl_rate")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "pnl_rate": fin.expected_pnl_rate(delta, a, kappa, alpha, p, v_h, v_l)
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_market_making_breakeven_alpha(
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
        """Market-making / HFT quoting kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): breakeven alpha."""
        if ctx:
            await ctx.info("quant_market_making_breakeven_alpha")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {"breakeven_alpha": fin.breakeven_alpha(delta, p, v_h, v_l)}
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_microstructure_ofi_series(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Order-flow / toxicity / self-excitation kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): ofi series."""
        if ctx:
            await ctx.info("quant_microstructure_ofi_series")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
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
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_microstructure_microprice_series(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Order-flow / toxicity / self-excitation kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): microprice series."""
        if ctx:
            await ctx.info("quant_microstructure_microprice_series")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "microprice": fin.microprice_series(
                    _loads("bid_px_json", bid_px_json),
                    _loads("bid_sz_json", bid_sz_json),
                    _loads("ask_px_json", ask_px_json),
                    _loads("ask_sz_json", ask_sz_json),
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_microstructure_vpin_pm(
        buy_vol_json: str = Field(
            default="[]", description="JSON list of bucket buy volumes (vpin_pm)."
        ),
        sell_vol_json: str = Field(
            default="[]", description="JSON list of bucket sell volumes (vpin_pm)."
        ),
        p_mean_json: str = Field(
            default="[]", description="JSON list of bucket mean prices (vpin_pm)."
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Order-flow / toxicity / self-excitation kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): vpin pm."""
        if ctx:
            await ctx.info("quant_microstructure_vpin_pm")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "vpin": fin.vpin_pm(
                    _loads("buy_vol_json", buy_vol_json),
                    _loads("sell_vol_json", sell_vol_json),
                    _loads("p_mean_json", p_mean_json),
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_microstructure_hawkes_mle(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Order-flow / toxicity / self-excitation kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): hawkes mle."""
        if ctx:
            await ctx.info("quant_microstructure_hawkes_mle")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.hawkes_mle(_loads("times_json", times_json), t_horizon, max_iter)
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_microstructure_hardiman_bouchaud(
        times_json: str = Field(
            default="[]",
            description="JSON list of event times (hawkes_mle/hardiman_bouchaud).",
        ),
        t_horizon: float = Field(
            default=1.0,
            description="Observation horizon (hawkes_mle/hardiman_bouchaud).",
        ),
        n_windows: int = Field(
            default=100, description="Count windows (hardiman_bouchaud)."
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Order-flow / toxicity / self-excitation kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): hardiman bouchaud."""
        if ctx:
            await ctx.info("quant_microstructure_hardiman_bouchaud")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "branching_ratio": fin.hardiman_bouchaud(
                    _loads("times_json", times_json), t_horizon, n_windows
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_microstructure_queue_imbalance(
        bid_q_json: str = Field(
            default="[]",
            description="JSON list of best-bid queue sizes (queue_imbalance).",
        ),
        ask_q_json: str = Field(
            default="[]",
            description="JSON list of best-ask queue sizes (queue_imbalance).",
        ),
        bid_rate_json: str = Field(
            default="[]",
            description="JSON list of bid-side arrival rates (queue_imbalance; uniform if empty).",
        ),
        ask_rate_json: str = Field(
            default="[]",
            description="JSON list of ask-side arrival rates (queue_imbalance; uniform if empty).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Order-flow / toxicity / self-excitation kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): queue imbalance."""
        if ctx:
            await ctx.info("quant_microstructure_queue_imbalance")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.queue_imbalance(
                _loads("bid_q_json", bid_q_json),
                _loads("ask_q_json", ask_q_json),
                _loads("bid_rate_json", bid_rate_json),
                _loads("ask_rate_json", ask_rate_json),
            )
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_microstructure_realized_vol_tick(
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
        """Order-flow / toxicity / self-excitation kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest): realized vol tick."""
        if ctx:
            await ctx.info("quant_microstructure_realized_vol_tick")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "realized_vol": fin.realized_vol_tick(
                    _loads("mid_json", mid_json), window
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_sizing_kelly_fraction(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Position-sizing kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): kelly fraction."""
        if ctx:
            await ctx.info("quant_sizing_kelly_fraction")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {"fraction": fin.kelly_fraction(q, c, fraction)}
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_sizing_bayesian_kelly(
        c: float = Field(
            default=0.5,
            description="Contract cost / price (kelly_fraction/bayesian_kelly).",
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Position-sizing kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): bayesian kelly."""
        if ctx:
            await ctx.info("quant_sizing_bayesian_kelly")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {"fraction": fin.bayesian_kelly(alpha, beta, c, n_quadrature)}
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_sizing_posterior_credible_interval(
        alpha: float = Field(
            default=1.0,
            description="Beta posterior α (bayesian_kelly/credible_interval).",
        ),
        beta: float = Field(
            default=1.0,
            description="Beta posterior β (bayesian_kelly/credible_interval).",
        ),
        level: float = Field(
            default=0.05,
            description="Tail level for the credible interval (e.g. 0.05 → 95%).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Position-sizing kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): posterior credible interval."""
        if ctx:
            await ctx.info("quant_sizing_posterior_credible_interval")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.posterior_credible_interval(alpha, beta, level)
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_sizing_empirical_kelly(
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
        """Position-sizing kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): empirical kelly."""
        if ctx:
            await ctx.info("quant_sizing_empirical_kelly")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "fraction": fin.empirical_kelly(
                    p,
                    b,
                    _loads("historical_returns_json", historical_returns_json),
                    n_simulations,
                    seed,
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_validation_purged_cpcv(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Backtest-validation / calibration kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): purged cpcv."""
        if ctx:
            await ctx.info("quant_validation_purged_cpcv")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "splits": fin.purged_cpcv(
                    n_samples, n_groups, n_test_groups, purge_window, embargo
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_validation_deflated_sharpe(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Backtest-validation / calibration kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): deflated sharpe."""
        if ctx:
            await ctx.info("quant_validation_deflated_sharpe")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "deflated_sharpe": fin.deflated_sharpe(
                    observed_sr,
                    n_trials,
                    _loads("sr_returns_json", sr_returns_json),
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_validation_probability_backtest_overfit(
        insample_json: str = Field(
            default="[]",
            description="JSON 2D matrix rows=splits, cols=strategies (PBO).",
        ),
        oos_json: str = Field(
            default="[]",
            description="JSON 2D matrix of out-of-sample performances (PBO).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Backtest-validation / calibration kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): probability backtest overfit."""
        if ctx:
            await ctx.info("quant_validation_probability_backtest_overfit")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "pbo": fin.probability_backtest_overfit(
                    _loads("insample_json", insample_json),
                    _loads("oos_json", oos_json),
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_validation_diebold_mariano(
        losses_a_json: str = Field(
            default="[]", description="JSON list of model-A losses (diebold_mariano)."
        ),
        losses_b_json: str = Field(
            default="[]", description="JSON list of model-B losses (diebold_mariano)."
        ),
        h: int = Field(
            default=1, description="Forecast horizon for HAC (diebold_mariano)."
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Backtest-validation / calibration kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): diebold mariano."""
        if ctx:
            await ctx.info("quant_validation_diebold_mariano")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.diebold_mariano(
                _loads("losses_a_json", losses_a_json),
                _loads("losses_b_json", losses_b_json),
                h,
            )
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_validation_brier_score(
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
        """Backtest-validation / calibration kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i): brier score."""
        if ctx:
            await ctx.info("quant_validation_brier_score")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "brier_score": fin.brier_score(
                    _loads("forecasts_json", forecasts_json),
                    _loads("outcomes_json", outcomes_json),
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_statespace_kalman_filter(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """State-space / statistical-arbitrage kernels (CONCEPT:EG-KG.domains.state-space-statistical-arbitrage): kalman filter."""
        if ctx:
            await ctx.info("quant_statespace_kalman_filter")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.kalman_filter_1d(
                _loads("observations_json", observations_json),
                f,
                q,
                h,
                r,
                x0,
                p0,
            )
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_statespace_kalman_beta(
        q: float = Field(
            default=1e-5,
            description="Process-noise variance (kalman_filter/kalman_beta).",
        ),
        r: float = Field(
            default=1e-3,
            description="Observation-noise variance (kalman_filter/kalman_beta).",
        ),
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """State-space / statistical-arbitrage kernels (CONCEPT:EG-KG.domains.state-space-statistical-arbitrage): kalman beta."""
        if ctx:
            await ctx.info("quant_statespace_kalman_beta")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.kalman_beta(
                _loads("market_returns_json", market_returns_json),
                _loads("asset_returns_json", asset_returns_json),
                q,
                r,
                beta0,
                p0,
            )
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_statespace_kalman_volatility(
        p0: float = Field(
            default=1.0,
            description="Initial state variance (kalman_filter/beta/volatility).",
        ),
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """State-space / statistical-arbitrage kernels (CONCEPT:EG-KG.domains.state-space-statistical-arbitrage): kalman volatility."""
        if ctx:
            await ctx.info("quant_statespace_kalman_volatility")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
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
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_statespace_adf_test(
        series_json: str = Field(
            default="[]",
            description="JSON list — the test series (adf_test).",
        ),
        max_lag: int = Field(default=1, description="Max ADF lag (adf_test)."),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """State-space / statistical-arbitrage kernels (CONCEPT:EG-KG.domains.state-space-statistical-arbitrage): adf test."""
        if ctx:
            await ctx.info("quant_statespace_adf_test")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.adf_test(_loads("series_json", series_json), max_lag)
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_statespace_ou_calibrate(
        spread_json: str = Field(
            default="[]",
            description="JSON list — the mean-reverting spread (ou_calibrate).",
        ),
        dt: float = Field(default=1.0, description="Sampling interval (ou_calibrate)."),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """State-space / statistical-arbitrage kernels (CONCEPT:EG-KG.domains.state-space-statistical-arbitrage): ou calibrate."""
        if ctx:
            await ctx.info("quant_statespace_ou_calibrate")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.ou_calibrate(_loads("spread_json", spread_json), dt)
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_statespace_ou_optimal_thresholds(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """State-space / statistical-arbitrage kernels (CONCEPT:EG-KG.domains.state-space-statistical-arbitrage): ou optimal thresholds."""
        if ctx:
            await ctx.info("quant_statespace_ou_optimal_thresholds")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.ou_optimal_thresholds(theta, mu, sigma, sigma_eq, cost)
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_statespace_markov_transition(
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
        """State-space / statistical-arbitrage kernels (CONCEPT:EG-KG.domains.state-space-statistical-arbitrage): markov transition."""
        if ctx:
            await ctx.info("quant_statespace_markov_transition")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "transition_matrix": fin.markov_transition_matrix(
                    _loads("states_json", states_json), n_states
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_signals_order_book_imbalance(
        v_bid_json: str = Field(
            default="[]",
            description="JSON list of bid volumes (order_book_imbalance).",
        ),
        v_ask_json: str = Field(
            default="[]",
            description="JSON list of ask volumes (order_book_imbalance).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Signal-combination / breadth kernels (CONCEPT:EG-KG.domains.quant-finance): order book imbalance."""
        if ctx:
            await ctx.info("quant_signals_order_book_imbalance")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "imbalance": fin.order_book_imbalance(
                    _loads("v_bid_json", v_bid_json),
                    _loads("v_ask_json", v_ask_json),
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_signals_information_ratio(
        ic: float = Field(
            default=0.0,
            description="Information coefficient per signal (information_ratio).",
        ),
        n_independent: float = Field(
            default=1.0,
            description="Effective independent breadth (information_ratio).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Signal-combination / breadth kernels (CONCEPT:EG-KG.domains.quant-finance): information ratio."""
        if ctx:
            await ctx.info("quant_signals_information_ratio")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {"information_ratio": fin.information_ratio(ic, n_independent)}
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_signals_effective_independent_n(
        returns_matrix_json: str = Field(
            default="[]",
            description=(
                "JSON 2D matrix rows=signals/assets, cols=time "
                "(effective_independent_n/alpha_combination)."
            ),
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Signal-combination / breadth kernels (CONCEPT:EG-KG.domains.quant-finance): effective independent n."""
        if ctx:
            await ctx.info("quant_signals_effective_independent_n")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "effective_n": fin.effective_independent_n(
                    _loads("returns_matrix_json", returns_matrix_json)
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_signals_alpha_combination(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Signal-combination / breadth kernels (CONCEPT:EG-KG.domains.quant-finance): alpha combination."""
        if ctx:
            await ctx.info("quant_signals_alpha_combination")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {
                "weights": fin.alpha_combination_engine(
                    _loads("returns_matrix_json", returns_matrix_json),
                    lookback,
                )
            }
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_signals_convergence_gate(
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
        """Signal-combination / breadth kernels (CONCEPT:EG-KG.domains.quant-finance): convergence gate."""
        if ctx:
            await ctx.info("quant_signals_convergence_gate")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.convergence_gate(
                _loads("strengths_json", strengths_json),
                strong_threshold,
                min_agree,
            )
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_signals_spread_reversion(
        bid_px_json: str = Field(
            default="[]", description="JSON list of bid prices (spread_reversion)."
        ),
        ask_px_json: str = Field(
            default="[]", description="JSON list of ask prices (spread_reversion)."
        ),
        window: int = Field(
            default=20,
            description="Rolling window for the spread z-score (spread_reversion).",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Signal-combination / breadth kernels (CONCEPT:EG-KG.domains.quant-finance): spread reversion."""
        if ctx:
            await ctx.info("quant_signals_spread_reversion")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.spread_reversion(
                _loads("bid_px_json", bid_px_json),
                _loads("ask_px_json", ask_px_json),
                window,
            )
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(
        tags={"quant"},
        annotations={
            "readOnlyHint": True,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    )
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
        """Forensic-accounting report (CONCEPT:EG-KG.domains.forensic-accounting-kernels).

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
        except ValueError:
            return {"error": "Operation failed"}
        try:
            return fin.forensic_report(this_year, prior_year)
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_derivatives_implied_vol(
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """SABR stochastic-volatility surface kernels (CONCEPT:AU-KG.domains.derivatives): implied vol."""
        if ctx:
            await ctx.info("quant_derivatives_implied_vol")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return {"implied_vol": fin.sabr_implied_vol(f, k, t, alpha, beta, rho, nu)}
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_derivatives_smile(
        f: float = Field(default=0.0, description="Forward / underlying price."),
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
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """SABR stochastic-volatility surface kernels (CONCEPT:AU-KG.domains.derivatives): smile."""
        if ctx:
            await ctx.info("quant_derivatives_smile")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
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
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"quant"})
    async def quant_derivatives_calibrate(
        f: float = Field(default=0.0, description="Forward / underlying price."),
        t: float = Field(default=0.0, description="Time to expiry in years."),
        beta: float = Field(
            default=0.5,
            description="SABR β — elasticity (fixed for calibrate, default 0.5).",
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
        """SABR stochastic-volatility surface kernels (CONCEPT:AU-KG.domains.derivatives): calibrate."""
        if ctx:
            await ctx.info("quant_derivatives_calibrate")
        fin = _finance()
        if fin is None:
            return {"error": _ENGINE_REQUIRED_ERR}
        try:
            return fin.sabr_calibrate(
                f,
                t,
                _loads("strikes_json", strikes_json),
                _loads("market_vols_json", market_vols_json),
                beta,
            )
        except Exception:
            return {"error": "Operation failed"}
