import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import attack4  # noqa: E402


def _state(node_id: int) -> np.ndarray:
    scs = np.zeros((attack4.N, 3), dtype=np.float64)
    scs[:, 0] = float(node_id)
    return scs


def _install_graph_mocks(monkeypatch, graph: dict[int, list[tuple[str, int, float]]]):
    def fake_stage_b(
        root_scs,
        root_cx,
        root_cy,
        root_R,
        base_feats,
        base_wit,
        current_best_R,
        workers=1,
        maxiter=300,
        margin_pair=2e-6,
        margin_contain=2e-6,
        include_pairs=False,
        log=print,
    ):
        node_id = int(root_scs[0, 0])
        results = []
        for kind, child_id, rounded_R in graph.get(node_id, []):
            results.append(
                dict(
                    kind=kind,
                    move=dict(kind=kind, _type="mock", child=child_id),
                    rounded_R=rounded_R,
                    unrounded_R=rounded_R,
                    overlaps=0,
                    viol=0.0,
                    scs=_state(child_id),
                )
            )
        results.sort(key=lambda item: item["rounded_R"])
        return results

    def fake_extract_contact_graph(scs, cx, cy, R):
        node_id = int(scs[0, 0])
        feats = [attack4.Feature("CONTAIN_ARC", i=0, note=f"node={node_id}")]
        witnesses = [(0, 1)] if node_id % 2 == 1 else []
        return feats, witnesses

    monkeypatch.setattr(attack4, "stage_b", fake_stage_b)
    monkeypatch.setattr(attack4, "extract_contact_graph", fake_extract_contact_graph)
    monkeypatch.setattr(
        attack4.geom, "rnd", lambda scs: np.asarray(scs, dtype=np.float64)
    )
    monkeypatch.setattr(
        attack4.geom,
        "mec_info",
        lambda scs: (0.0, 0.0, float(100.0 - float(scs[0, 0]))),
    )


def test_stage_b_lookahead_recovers_known_improver(monkeypatch):
    graph = {
        0: [("root->a", 1, 10.2), ("root->b", 2, 10.4)],
        1: [("a->goal", 3, 8.8), ("a->worse", 4, 10.1)],
        2: [("b->dead", 5, 10.6)],
    }
    _install_graph_mocks(monkeypatch, graph)
    state0 = _state(0)

    assert (
        attack4.stage_b_lookahead(
            state0,
            0.0,
            0.0,
            9.5,
            [],
            [],
            9.5,
            max_depth=1,
            beam_width=5,
            log=lambda _msg: None,
        )
        is None
    )

    result = attack4.stage_b_lookahead(
        state0,
        0.0,
        0.0,
        9.5,
        [],
        [],
        9.5,
        max_depth=2,
        beam_width=5,
        log=lambda _msg: None,
    )

    assert result is not None
    assert result["depth"] == 2
    assert [move["kind"] for move in result["chain"]] == ["root->a", "a->goal"]
    assert result["chain_Rs"] == [10.2, 8.8]
    assert result["final_R"] == 8.8
    assert int(result["final_scs"][0, 0]) == 3
    assert [feat.note for feat in result["final_feats"]] == ["node=3"]
    assert result["final_wit"] == [(0, 1)]


def test_stage_b_lookahead_default_off(tmp_path: Path, monkeypatch):
    state0 = _state(0)
    stage_a_result = dict(
        scs=state0.copy(),
        cx=1.0,
        cy=2.0,
        R=6.0,
        success=True,
        message="ok",
        nit=0,
        constr_violation=0.0,
    )

    monkeypatch.setattr(
        attack4, "load_seed", lambda path: (state0.copy(), 1.0, 2.0, 6.0)
    )
    monkeypatch.setattr(
        attack4, "extract_contact_graph", lambda scs, cx, cy, R: ([], [])
    )
    monkeypatch.setattr(attack4, "stage_a", lambda *args, **kwargs: stage_a_result)
    monkeypatch.setattr(attack4, "validate", lambda scs: 5.4321)
    monkeypatch.setattr(
        attack4,
        "stage_b",
        lambda *args, **kwargs: [
            dict(
                kind="noop",
                move=dict(kind="noop", _type="mock"),
                rounded_R=5.4321,
                unrounded_R=5.4321,
                overlaps=0,
                viol=0.0,
                scs=state0.copy(),
            )
        ],
    )

    def unexpected_lookahead(*args, **kwargs):
        raise AssertionError("stage_b_lookahead should not run when disabled")

    monkeypatch.setattr(attack4, "stage_b_lookahead", unexpected_lookahead)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "attack4.py",
            "--root",
            "pool/best.json",
            "--stage-b",
            "--depth",
            "1",
            "--lookahead-depth",
            "1",
            "--out",
            str(tmp_path),
        ],
    )

    attack4.main()

    with open(tmp_path / "summary.json") as handle:
        summary = json.load(handle)
    assert summary == {"best_R": 5.4321, "history": [{"stage": "A", "R": 5.4321}]}
    assert (tmp_path / "stage_b_depth1.json").exists()
    assert not (tmp_path / "lookahead_d1.json").exists()


def test_stage_b_lookahead_respects_beam(monkeypatch):
    graph = {
        0: [
            ("root->1", 1, 10.0),
            ("root->2", 2, 10.1),
            ("root->3", 3, 10.2),
            ("root->4", 4, 10.3),
            ("root->5", 5, 10.4),
        ],
        1: [("1->dead", 11, 10.6)],
        2: [("2->dead", 12, 10.7)],
        3: [("3->dead", 13, 10.8)],
        4: [("4->dead", 14, 10.9)],
        5: [("5->goal", 15, 8.7)],
    }
    _install_graph_mocks(monkeypatch, graph)
    state0 = _state(0)
    call_order: list[int] = []
    original_stage_b = attack4.stage_b

    def counting_stage_b(*args, **kwargs):
        call_order.append(int(args[0][0, 0]))
        return original_stage_b(*args, **kwargs)

    monkeypatch.setattr(attack4, "stage_b", counting_stage_b)
    result_beam_1 = attack4.stage_b_lookahead(
        state0,
        0.0,
        0.0,
        9.5,
        [],
        [],
        9.5,
        max_depth=2,
        beam_width=1,
        log=lambda _msg: None,
    )
    calls_beam_1 = list(call_order)

    call_order.clear()
    result_beam_5 = attack4.stage_b_lookahead(
        state0,
        0.0,
        0.0,
        9.5,
        [],
        [],
        9.5,
        max_depth=2,
        beam_width=5,
        log=lambda _msg: None,
    )
    calls_beam_5 = list(call_order)

    assert result_beam_1 is None
    assert result_beam_5 is not None
    assert [move["kind"] for move in result_beam_5["chain"]] == ["root->5", "5->goal"]
    assert len(calls_beam_1) < len(calls_beam_5)
    assert calls_beam_1 == [0, 1]
    assert calls_beam_5 == [0, 1, 2, 3, 4, 5]
