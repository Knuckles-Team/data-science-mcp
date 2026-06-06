"""Tests for the quant MCP tools that expose epistemic-graph finance kernels.

Compute runs in the Rust epistemic-graph engine over ``client.finance.*``, so the
end-to-end tool calls depend on a reachable engine and use the shared
``require_engine`` fixture (cleanly skipped when no engine socket is present).
The engine-independent assertions (registration, action routing, graceful error
when the engine is down) run unconditionally.
"""

import sys

# Set dummy sys.argv before importing anything to prevent create_mcp_server
# parsing the pytest CLI.
sys.argv = ["mcp_server.py"]

import json

import pytest

from data_science_mcp.ml_engine import MLEngine, _UNPROBED
from data_science_mcp.mcp_server import get_mcp_instance

QUANT_TOOLS = {
    "quant_market_making",
    "quant_microstructure",
    "quant_sizing",
    "quant_validation",
    "quant_forensic",
    "quant_statespace",
    "quant_signals",
    "quant_derivatives",
}


def _skip_if_unknown_variant(res: dict) -> None:
    """Skip when the live engine daemon is a version behind and lacks the kernel.

    The KG-2.20h/i kernels may not exist on an older engine build; the engine
    surfaces that as an 'unknown variant' MessagePack error. Mirror the
    no-socket skip so engine-backed tests never hard-fail on a stale daemon.
    """
    err = str(res.get("error", "")).lower()
    if "unknown variant" in err or "unknown method" in err:
        pytest.skip(f"engine lacks kernel (version behind): {res['error']}")


@pytest.mark.asyncio
async def test_quant_tools_registered():
    """All five action-routed quant tools are registered and tagged 'quant'."""
    mcp, _, _, registered_tags = get_mcp_instance()
    assert "quant" in registered_tags
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert QUANT_TOOLS <= names
    for t in tools:
        if t.name in QUANT_TOOLS:
            assert "quant" in t.tags


@pytest.mark.asyncio
async def test_quant_unknown_action_errors(require_engine):
    """An unrecognized action returns a clean error (no exception)."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    mm = next(t for t in tools if t.name == "quant_market_making")
    res = await mm.fn(action="does_not_exist", ctx=None)
    assert "error" in res
    assert "Unknown action" in res["error"]


@pytest.mark.asyncio
async def test_quant_engine_unavailable_graceful(monkeypatch):
    """With no engine reachable, tools degrade to an error dict (never raise)."""
    monkeypatch.setattr(MLEngine, "_rust_client_cache", None, raising=False)
    monkeypatch.setattr(MLEngine, "_rust_client", classmethod(lambda cls: None))

    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    sizing = next(t for t in tools if t.name == "quant_sizing")
    res = await sizing.fn(action="kelly_fraction", q=0.6, c=0.5, ctx=None)
    assert "error" in res

    # restore probe sentinel for subsequent tests
    MLEngine._rust_client_cache = _UNPROBED


@pytest.mark.asyncio
async def test_quant_invalid_json_errors(require_engine):
    """Malformed JSON list arguments surface a clean error."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    micro = next(t for t in tools if t.name == "quant_microstructure")
    res = await micro.fn(action="vpin_pm", buy_vol_json="{bad", ctx=None)
    assert "error" in res


@pytest.mark.asyncio
async def test_quant_market_making(require_engine):
    """Avellaneda-Stoikov, GLT, logit, and the scalar GM/PnL kernels run."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    mm = next(t for t in tools if t.name == "quant_market_making")

    res_as = await mm.fn(
        action="avellaneda_stoikov",
        mid=0.5,
        inventory=0.0,
        sigma=0.1,
        gamma=0.1,
        kappa=1.5,
        tau=1.0,
        ctx=None,
    )
    assert "error" not in res_as
    assert {"bid", "ask", "reservation", "half_spread", "withdraw"} <= set(res_as)

    res_glt = await mm.fn(
        action="glt_quotes",
        mid=0.5,
        inventory=0.0,
        sigma=0.1,
        gamma=0.1,
        kappa=1.5,
        a=1.0,
        ctx=None,
    )
    assert "error" not in res_glt and "bid" in res_glt

    res_logit = await mm.fn(
        action="logit_quotes",
        p_mid=0.5,
        inventory=0.0,
        sigma=0.1,
        gamma=0.1,
        kappa=1.5,
        tau=1.0,
        boundary_m=0.0,
        ctx=None,
    )
    assert "error" not in res_logit and "withdraw" in res_logit

    res_gm = await mm.fn(action="glosten_milgrom_spread", alpha=0.2, p=0.5, ctx=None)
    assert "spread" in res_gm and isinstance(res_gm["spread"], (int, float))

    res_pnl = await mm.fn(
        action="expected_pnl_rate",
        delta=0.05,
        a=1.0,
        kappa=1.5,
        alpha=0.2,
        p=0.5,
        v_h=1.0,
        v_l=0.0,
        ctx=None,
    )
    assert "pnl_rate" in res_pnl

    res_be = await mm.fn(
        action="breakeven_alpha", delta=0.05, p=0.5, v_h=1.0, v_l=0.0, ctx=None
    )
    assert "breakeven_alpha" in res_be


@pytest.mark.asyncio
async def test_quant_microstructure(require_engine):
    """OFI / microprice / VPIN / Hawkes kernels run end-to-end."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    micro = next(t for t in tools if t.name == "quant_microstructure")

    ts = list(range(6))
    bid_px = [0.49, 0.49, 0.50, 0.50, 0.51, 0.51]
    bid_sz = [100.0, 120.0, 90.0, 110.0, 130.0, 95.0]
    ask_px = [0.51, 0.51, 0.52, 0.52, 0.53, 0.53]
    ask_sz = [100.0, 80.0, 95.0, 105.0, 90.0, 100.0]

    res_ofi = await micro.fn(
        action="ofi_series",
        ts_json=json.dumps(ts),
        bid_px_json=json.dumps(bid_px),
        bid_sz_json=json.dumps(bid_sz),
        ask_px_json=json.dumps(ask_px),
        ask_sz_json=json.dumps(ask_sz),
        window_secs=1.0,
        ctx=None,
    )
    assert "error" not in res_ofi and isinstance(res_ofi["ofi"], list)

    res_mp = await micro.fn(
        action="microprice_series",
        bid_px_json=json.dumps(bid_px),
        bid_sz_json=json.dumps(bid_sz),
        ask_px_json=json.dumps(ask_px),
        ask_sz_json=json.dumps(ask_sz),
        ctx=None,
    )
    assert "error" not in res_mp and isinstance(res_mp["microprice"], list)

    res_vpin = await micro.fn(
        action="vpin_pm",
        buy_vol_json=json.dumps([50.0, 60.0, 40.0]),
        sell_vol_json=json.dumps([50.0, 40.0, 60.0]),
        p_mean_json=json.dumps([0.5, 0.5, 0.5]),
        ctx=None,
    )
    assert "vpin" in res_vpin

    times = [0.1, 0.3, 0.35, 0.9, 1.2, 1.25, 1.8, 2.4]
    res_hawkes = await micro.fn(
        action="hawkes_mle",
        times_json=json.dumps(times),
        t_horizon=3.0,
        max_iter=200,
        ctx=None,
    )
    assert "error" not in res_hawkes
    assert "branching_ratio" in res_hawkes

    res_hb = await micro.fn(
        action="hardiman_bouchaud",
        times_json=json.dumps(times),
        t_horizon=3.0,
        n_windows=10,
        ctx=None,
    )
    assert "branching_ratio" in res_hb


@pytest.mark.asyncio
async def test_quant_sizing(require_engine):
    """Kelly / Bayesian-Kelly / credible-interval kernels run."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    sizing = next(t for t in tools if t.name == "quant_sizing")

    res_k = await sizing.fn(
        action="kelly_fraction", q=0.6, c=0.5, fraction=0.25, ctx=None
    )
    assert "fraction" in res_k

    res_bk = await sizing.fn(
        action="bayesian_kelly", alpha=6.0, beta=4.0, c=0.5, n_quadrature=50, ctx=None
    )
    assert "fraction" in res_bk

    res_ci = await sizing.fn(
        action="posterior_credible_interval", alpha=6.0, beta=4.0, level=0.05, ctx=None
    )
    assert "error" not in res_ci
    assert "lower" in res_ci and "upper" in res_ci


@pytest.mark.asyncio
async def test_quant_validation(require_engine):
    """CPCV / deflated-Sharpe / PBO / Diebold-Mariano kernels run."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    val = next(t for t in tools if t.name == "quant_validation")

    res_cpcv = await val.fn(
        action="purged_cpcv",
        n_samples=60,
        n_groups=6,
        n_test_groups=2,
        purge_window=1,
        embargo=1,
        ctx=None,
    )
    assert "error" not in res_cpcv
    assert isinstance(res_cpcv["splits"], list) and res_cpcv["splits"]
    assert {"train", "test"} <= set(res_cpcv["splits"][0])

    res_dsr = await val.fn(
        action="deflated_sharpe",
        observed_sr=1.5,
        n_trials=20,
        sr_returns_json=json.dumps([0.01, -0.02, 0.03, 0.0, 0.015, -0.01] * 10),
        ctx=None,
    )
    assert "deflated_sharpe" in res_dsr

    res_pbo = await val.fn(
        action="probability_backtest_overfit",
        insample_json=json.dumps([[0.1, 0.2], [0.15, 0.05], [0.2, 0.1], [0.05, 0.25]]),
        oos_json=json.dumps([[0.08, 0.18], [0.12, 0.04], [0.18, 0.09], [0.03, 0.2]]),
        ctx=None,
    )
    assert "pbo" in res_pbo

    res_dm = await val.fn(
        action="diebold_mariano",
        losses_a_json=json.dumps([0.1, 0.2, 0.15, 0.12, 0.18, 0.11]),
        losses_b_json=json.dumps([0.12, 0.19, 0.17, 0.15, 0.2, 0.13]),
        h=1,
        ctx=None,
    )
    assert "error" not in res_dm
    assert {"statistic", "p_value", "a_better"} <= set(res_dm)


@pytest.mark.asyncio
async def test_quant_forensic(require_engine):
    """forensic_report returns the full Beneish/Altman/Piotroski/Sloan summary."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    forensic = next(t for t in tools if t.name == "quant_forensic")

    def year(scale: float) -> dict:
        return {
            "sales": 1000.0 * scale,
            "cogs": 600.0 * scale,
            "sga": 150.0 * scale,
            "net_income": 120.0 * scale,
            "cfo": 110.0 * scale,
            "receivables": 200.0 * scale,
            "current_assets": 500.0 * scale,
            "current_liabilities": 300.0 * scale,
            "ppe_net": 400.0 * scale,
            "depreciation": 50.0 * scale,
            "total_assets": 1500.0 * scale,
            "total_liabilities": 700.0 * scale,
            "long_term_debt": 400.0 * scale,
            "retained_earnings": 300.0 * scale,
            "ebit": 180.0 * scale,
            "market_cap": 2000.0 * scale,
            "shares": 100.0,
        }

    res = await forensic.fn(
        this_year_json=json.dumps(year(1.2)),
        prior_year_json=json.dumps(year(1.0)),
        ctx=None,
    )
    assert "error" not in res
    assert {
        "m_score",
        "z_score",
        "f_score",
        "accruals_ratio",
        "flags",
        "verdict",
    } <= set(res)


@pytest.mark.asyncio
async def test_quant_sizing_empirical_kelly(require_engine):
    """Empirical (Monte-Carlo) Kelly fraction runs via the engine."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    sizing = next(t for t in tools if t.name == "quant_sizing")
    res = await sizing.fn(
        action="empirical_kelly",
        p=0.55,
        b=1.0,
        historical_returns_json=json.dumps([1.0, -1.0, 1.0, 1.0, -1.0, 1.0] * 5),
        n_simulations=2000,
        seed=42,
        ctx=None,
    )
    _skip_if_unknown_variant(res)
    assert "error" not in res
    assert "fraction" in res and isinstance(res["fraction"], (int, float))


@pytest.mark.asyncio
async def test_quant_validation_brier(require_engine):
    """Brier calibration score runs via the engine."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    val = next(t for t in tools if t.name == "quant_validation")
    res = await val.fn(
        action="brier_score",
        forecasts_json=json.dumps([0.9, 0.1, 0.8, 0.3, 0.6]),
        outcomes_json=json.dumps([1, 0, 1, 0, 1]),
        ctx=None,
    )
    _skip_if_unknown_variant(res)
    assert "error" not in res
    assert "brier_score" in res


@pytest.mark.asyncio
async def test_quant_statespace(require_engine):
    """Kalman / ADF / OU / Markov state-space kernels run end-to-end."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    ss = next(t for t in tools if t.name == "quant_statespace")

    obs = [float(i) + (0.1 if i % 2 else -0.1) for i in range(30)]
    res_kf = await ss.fn(
        action="kalman_filter",
        observations_json=json.dumps(obs),
        f=1.0,
        q=1e-5,
        h=1.0,
        r=1e-3,
        x0=0.0,
        p0=1.0,
        ctx=None,
    )
    _skip_if_unknown_variant(res_kf)
    assert "error" not in res_kf
    assert "states" in res_kf and "variances" in res_kf

    mkt = [0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.012] * 4
    ast = [0.012, -0.025, 0.02, 0.004, -0.013, 0.025, -0.006, 0.015] * 4
    res_kb = await ss.fn(
        action="kalman_beta",
        market_returns_json=json.dumps(mkt),
        asset_returns_json=json.dumps(ast),
        q=1e-5,
        r=1e-3,
        beta0=1.0,
        p0=1.0,
        ctx=None,
    )
    _skip_if_unknown_variant(res_kb)
    assert "error" not in res_kb and "states" in res_kb

    res_kv = await ss.fn(
        action="kalman_volatility",
        returns_json=json.dumps(mkt),
        q_vol=0.1,
        r_vol=1.0,
        log_var0=None,
        p0=1.0,
        annualization=252.0,
        ctx=None,
    )
    _skip_if_unknown_variant(res_kv)
    assert "error" not in res_kv and isinstance(res_kv["annualized_vol"], list)

    spread = [10.0 + (-1) ** i * 0.5 + 0.05 * i for i in range(40)]
    res_adf = await ss.fn(
        action="adf_test", series_json=json.dumps(spread), max_lag=1, ctx=None
    )
    _skip_if_unknown_variant(res_adf)
    assert "error" not in res_adf
    assert {"statistic", "stationary_5pct"} <= set(res_adf)

    ou_spread = []
    x = 0.0
    for i in range(60):
        x = 0.8 * x + ((-1) ** i) * 0.2
        ou_spread.append(x)
    res_ou = await ss.fn(
        action="ou_calibrate", spread_json=json.dumps(ou_spread), dt=1.0, ctx=None
    )
    _skip_if_unknown_variant(res_ou)
    assert "error" not in res_ou
    assert {"theta", "mu", "sigma", "half_life", "sigma_eq"} <= set(res_ou)

    res_thr = await ss.fn(
        action="ou_optimal_thresholds",
        theta=float(res_ou["theta"]),
        mu=float(res_ou["mu"]),
        sigma=float(res_ou["sigma"]),
        sigma_eq=float(res_ou["sigma_eq"]),
        cost=0.0,
        ctx=None,
    )
    _skip_if_unknown_variant(res_thr)
    assert "error" not in res_thr
    assert {"entry_long", "entry_short", "exit"} <= set(res_thr)

    res_mc = await ss.fn(
        action="markov_transition",
        states_json=json.dumps([0, 1, 1, 0, 1, 0, 0, 1, 1, 1]),
        n_states=2,
        ctx=None,
    )
    _skip_if_unknown_variant(res_mc)
    assert "error" not in res_mc
    assert isinstance(res_mc["transition_matrix"], list)


@pytest.mark.asyncio
async def test_quant_signals(require_engine):
    """Signal-combination / breadth kernels run end-to-end."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    sig = next(t for t in tools if t.name == "quant_signals")

    res_obi = await sig.fn(
        action="order_book_imbalance",
        v_bid_json=json.dumps([100.0, 120.0, 90.0]),
        v_ask_json=json.dumps([80.0, 110.0, 95.0]),
        ctx=None,
    )
    _skip_if_unknown_variant(res_obi)
    assert "error" not in res_obi and isinstance(res_obi["imbalance"], list)

    res_ir = await sig.fn(
        action="information_ratio", ic=0.05, n_independent=100.0, ctx=None
    )
    _skip_if_unknown_variant(res_ir)
    assert "error" not in res_ir and "information_ratio" in res_ir

    matrix = [
        [0.01, -0.02, 0.03, 0.0, 0.015, -0.01, 0.02, -0.005],
        [0.012, -0.018, 0.025, 0.002, 0.013, -0.012, 0.018, -0.004],
        [-0.01, 0.02, -0.03, 0.0, -0.015, 0.01, -0.02, 0.005],
    ]
    res_en = await sig.fn(
        action="effective_independent_n",
        returns_matrix_json=json.dumps(matrix),
        ctx=None,
    )
    _skip_if_unknown_variant(res_en)
    assert "error" not in res_en and "effective_n" in res_en

    res_ac = await sig.fn(
        action="alpha_combination",
        returns_matrix_json=json.dumps(matrix),
        lookback=4,
        ctx=None,
    )
    _skip_if_unknown_variant(res_ac)
    assert "error" not in res_ac and isinstance(res_ac["weights"], list)

    res_cg = await sig.fn(
        action="convergence_gate",
        strengths_json=json.dumps([0.7, 0.8, 0.65, 0.9, 0.72, -0.1]),
        strong_threshold=0.6,
        min_agree=5,
        ctx=None,
    )
    _skip_if_unknown_variant(res_cg)
    assert "error" not in res_cg
    assert {"agree", "total", "fraction", "direction", "pass"} <= set(res_cg)


@pytest.mark.asyncio
async def test_quant_statespace_invalid_json():
    """Malformed JSON to a state-space action returns a clean error (no engine)."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    ss = next(t for t in tools if t.name == "quant_statespace")
    res = await ss.fn(action="adf_test", series_json="{bad", ctx=None)
    assert "error" in res


@pytest.mark.asyncio
async def test_quant_forensic_invalid_json():
    """Malformed forensic input JSON returns a clean error (no engine needed)."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    forensic = next(t for t in tools if t.name == "quant_forensic")
    res = await forensic.fn(this_year_json="{bad", prior_year_json="{}", ctx=None)
    assert "error" in res


@pytest.mark.asyncio
async def test_quant_derivatives_unknown_action(require_engine):
    """An unrecognized SABR action returns a clean error (no exception)."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    deriv = next(t for t in tools if t.name == "quant_derivatives")
    res = await deriv.fn(action="nope", ctx=None)
    assert "error" in res


@pytest.mark.asyncio
async def test_quant_derivatives_invalid_json():
    """Malformed strikes JSON returns a clean error (no engine needed)."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    deriv = next(t for t in tools if t.name == "quant_derivatives")
    res = await deriv.fn(action="smile", strikes_json="{bad", ctx=None)
    assert "error" in res


@pytest.mark.asyncio
async def test_quant_derivatives_sabr(require_engine):
    """SABR implied_vol / smile / calibrate kernels run end-to-end."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    deriv = next(t for t in tools if t.name == "quant_derivatives")

    iv = await deriv.fn(
        action="implied_vol",
        f=1.0,
        k=1.0,
        t=1.0,
        alpha=0.2,
        beta=0.5,
        rho=-0.3,
        nu=0.4,
        ctx=None,
    )
    _skip_if_unknown_variant(iv)
    assert "error" not in iv
    assert iv["implied_vol"] > 0

    sm = await deriv.fn(
        action="smile",
        f=1.0,
        strikes_json="[0.8, 0.9, 1.0, 1.1, 1.2]",
        t=1.0,
        alpha=0.2,
        beta=0.5,
        rho=-0.3,
        nu=0.4,
        ctx=None,
    )
    _skip_if_unknown_variant(sm)
    assert "error" not in sm
    assert len(sm["vols"]) == 5

    cal = await deriv.fn(
        action="calibrate",
        f=1.0,
        t=1.0,
        strikes_json="[0.8, 0.9, 1.0, 1.1, 1.2]",
        market_vols_json=json.dumps(sm["vols"]),
        beta=0.5,
        ctx=None,
    )
    _skip_if_unknown_variant(cal)
    assert "error" not in cal
    assert {"alpha", "beta", "rho", "nu", "rmse", "converged"} <= set(cal)
