"""CacheRL cached-rollout + thinking-augmentation tests — CONCEPT:ML-013."""

from data_science_mcp.cache_agent_loop import (
    CacheAgentLoop,
    CacheTier,
    ThinkingTraceAugmenter,
    ThreeTierToolCache,
    augment_trajectory,
)


def test_exact_hit_after_first_store():
    c = ThreeTierToolCache()
    assert not c.lookup("search", {"q": "kyle model"}).hit
    c.put("search", {"q": "kyle model"}, "RESULT")
    res = c.lookup("search", {"q": "kyle model"})
    assert res.hit and res.tier is CacheTier.EXACT and res.value == "RESULT"
    # Argument order does not matter (canonicalized).
    assert c.lookup("search", {"q": "kyle model"}).value == "RESULT"


def test_fuzzy_hit_on_near_identical_args():
    c = ThreeTierToolCache(fuzzy_threshold=0.5)
    c.put("search", {"q": "alpha beta gamma delta"}, "R")
    res = c.lookup("search", {"q": "alpha beta gamma epsilon"})
    assert res.hit and res.tier is CacheTier.FUZZY and 0.0 < res.similarity < 1.0
    # A different tool never matches another tool's entry.
    assert not c.lookup("fetch", {"q": "alpha beta gamma delta"}).hit


def test_semantic_tier_with_embedder():
    # Two argument blobs share no tokens but identical embeddings ⇒ semantic hit.
    def embed(_s: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    c = ThreeTierToolCache(fuzzy_threshold=0.99, semantic_threshold=0.9, embed_fn=embed)
    c.put("lookup", {"a": "zzz"}, "R")
    res = c.lookup("lookup", {"b": "qqq"})
    assert res.hit and res.tier is CacheTier.SEMANTIC


def test_loop_serves_cache_and_counts_saved():
    calls = {"n": 0}

    def live(tool, args):
        calls["n"] += 1
        return f"obs:{tool}:{args}"

    loop = CacheAgentLoop(live_executor=live)
    v1, t1 = loop.call("search", {"q": "x"})
    v2, t2 = loop.call("search", {"q": "x"})  # exact cache hit, no live call
    v3, t3 = loop.call("search", {"q": "y"})  # miss → live
    assert t1 is CacheTier.LIVE and t2 is CacheTier.EXACT and t3 is CacheTier.LIVE
    assert v1 == v2
    assert calls["n"] == 2  # only two live executions for three calls
    assert loop.live_calls == 2 and loop.total_calls == 3
    assert loop.calls_saved() == 1
    assert loop.hit_rate() == round(1 / 3, 6)


def test_thinking_augmenter_interleaves_sources():
    steps = [
        {"tool": "search", "args": {"q": "kyle"}, "observation": "hit", "tier": CacheTier.FUZZY},
        {"tool": "read", "args": {"id": 3}, "observation": "doc", "tier": CacheTier.LIVE},
    ]
    aug = ThinkingTraceAugmenter(reason_fn=lambda step, hist: f"need {step['tool']}")
    segs = aug.augment(steps)
    sources = [s["source"] for s in segs]
    assert sources == ["model", "action", "observation", "model", "action", "observation"]
    # The thought explains the tool choice (why-this-tool).
    assert segs[0]["text"] == "need search"
    # Observation segments carry the cache tier that produced them.
    obs = [s for s in segs if s["source"] == "observation"]
    assert obs[0]["tier"] == "fuzzy" and obs[1]["tier"] == "live"


def test_augment_trajectory_wrapper_matches_class():
    steps = [{"tool": "t", "args": {}, "observation": "o"}]
    segs = augment_trajectory(steps, reason_fn=lambda s, h: "why")
    assert [s["source"] for s in segs] == ["model", "action", "observation"]
