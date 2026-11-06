import json
from pathlib import Path

from fastapi.testclient import TestClient

from console.api.main import app

_ROOT = Path(__file__).resolve().parents[2]
_P01 = _ROOT / "example" / "scenarios" / "P01_intro.json"


def test_p01_lists_ids_and_moves_orbits():
    data = json.loads(_P01.read_text(encoding="utf-8"))
    client = TestClient(app)
    loaded = client.post("/api/scenario", json=data)
    assert loaded.status_code == 200
    body = loaded.json()
    assert set(body["satellites"]) == {sat["id"] for sat in data["satellites"]}
    assert body["jobs"] == []
    assert body["scenario_id"] == data["meta"]["id"]

    at_0 = client.get("/api/orbits", params={"step": 0})
    at_10 = client.get("/api/orbits", params={"step": 10})
    assert at_0.status_code == 200
    assert at_10.status_code == 200
    sats0 = at_0.json()["satellites"]
    sats10 = at_10.json()["satellites"]
    assert len(sats0) == len(data["satellites"])
    assert len(sats10) == len(data["satellites"])
    assert [row["id"] for row in sats0] == [sat["id"] for sat in data["satellites"]]
    assert (sats0[0]["lat"], sats0[0]["lon"]) != (sats10[0]["lat"], sats10[0]["lon"])
    assert len(sats0[0]["orbit"]) == 64
    assert len(at_0.json()["sun_ecef"]) == 3


def test_tracks_stream_matches_steps():
    data = json.loads(_P01.read_text(encoding="utf-8"))
    client = TestClient(app)
    assert client.post("/api/scenario", json=data).status_code == 200
    raw = client.get("/api/tracks")
    assert raw.status_code == 200
    rows = [json.loads(line) for line in raw.text.splitlines() if line.strip()]
    frames = [row for row in rows if row["type"] == "frame"]
    assert [row["step"] for row in frames] == list(range(data["time"]["steps"] + 1))
    assert len(frames[0]["pos"]) == len(data["satellites"])
    assert frames[0]["pos"][0][:2] != frames[10]["pos"][0][:2]
    orbits = next(row for row in rows if row["type"] == "orbits")
    assert len(orbits["path"]) == len(data["satellites"])
    assert len(orbits["path"][0]) == 64

    at_0 = client.get("/api/orbits", params={"step": 0}).json()["satellites"][0]
    at_10 = client.get("/api/orbits", params={"step": 10}).json()["satellites"][0]
    assert abs(frames[0]["pos"][0][0] - at_0["lat"]) < 0.02
    assert abs(frames[0]["pos"][0][1] - at_0["lon"]) < 0.02
    assert abs(frames[10]["pos"][0][0] - at_10["lat"]) < 0.02
    assert abs(frames[10]["pos"][0][1] - at_10["lon"]) < 0.02


def test_dispatch_stream_without_checkpoint(monkeypatch):
    """Схема потока на жадном акторе: чекпоинт и сеть не грузятся."""
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from orbi.greedy import greedy_action

    import console.api.main as main_mod
    from console.api import dispatch as dispatch_mod

    def actor_for(scenario):
        limit = int(scenario["model"]["downlink_parallel_limit"])

        def actor(obs, _limit=limit):
            return greedy_action(obs, downlink_limit=_limit)

        return actor

    monkeypatch.setattr(dispatch_mod, "actor_for", actor_for)
    dispatch_mod.clear()
    main_mod._shift = None
    data = json.loads(_P01.read_text(encoding="utf-8"))
    client = TestClient(app)
    missing = client.get("/api/dispatch")
    assert missing.status_code == 409

    assert client.post("/api/scenario", json=data).status_code == 200
    first = client.get("/api/dispatch")
    assert first.status_code == 200
    second = client.get("/api/dispatch")
    assert second.text == first.text

    rows = [json.loads(line) for line in first.text.splitlines() if line.strip()]
    meta = rows[0]
    assert meta["type"] == "meta"
    assert meta["ids"] == [sat["id"] for sat in data["satellites"]]
    assert len(meta["jobs"]) == data["jobs"]["initial"]
    frames = [row for row in rows if row["type"] == "frame"]
    assert [row["step"] for row in frames] == list(range(data["time"]["steps"] + 1))
    assert frames[0]["job"] == []
    assert frames[0]["missed"] == []
    assert frames[0]["metrics"]["revenue_usd"] == 0
    assert all(len(row["sat"]) == len(meta["ids"]) for row in frames)
    assert all(len(cell) == 5 and cell[3] in (0, 1, 2) for row in frames for cell in row["sat"])
    assert all(isinstance(row.get("open"), list) for row in frames)
    assert all(len(row["take"]) == len(meta["ids"]) for row in frames)
    assert all(v in (0, 1) for row in frames for v in row["take"])
    n_jobs = len(meta["jobs"])
    assert all(0 <= i < n_jobs for row in frames for i in row["open"])
    if any(row["open"] for row in frames):
        assert any(row["take"].count(1) for row in frames if row["open"])
    done = [0] * len(meta["jobs"])
    for frame in frames:
        for index, _rem, ex, flag in frame["job"]:
            assert 0 <= index < len(meta["jobs"])
            assert ex == -1 or 0 <= ex < len(meta["ids"])
            done[index] = flag
        assert frame["metrics"]["blocked_command_count"] == 0
    revenue = sum(meta["jobs"][i]["value_usd"] for i, flag in enumerate(done) if flag)
    assert frames[-1]["metrics"]["jobs_completed"] == sum(done)
    assert abs(frames[-1]["metrics"]["revenue_usd"] - revenue) < 1e-4

    packed = client.get("/api/result")
    assert packed.status_code == 200
    body = packed.json()
    assert body["schema_version"] == "orbi-ops-result-1.0"
    assert body["summary"]["jobs_completed"] == frames[-1]["metrics"]["jobs_completed"]
    assert body["steps_executed"] == data["time"]["steps"]
    assert meta["jobs_seed"] == data["jobs"]["initial"]
    assert all(0 <= row["q"] <= 1 for row in frames)


def test_alt_dispatch_stream_without_checkpoint(monkeypatch):
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from orbi.greedy import greedy_action

    import console.api.main as main_mod
    from console.api import dispatch as dispatch_mod

    def actor_for(scenario):
        limit = int(scenario["model"]["downlink_parallel_limit"])

        def actor(obs, _limit=limit):
            return greedy_action(obs, downlink_limit=_limit)

        return actor

    monkeypatch.setattr(dispatch_mod, "actor_for", actor_for)
    monkeypatch.setattr(dispatch_mod, "actor_for_alt", actor_for)
    dispatch_mod.clear()
    main_mod._shift = None
    data = json.loads(_P01.read_text(encoding="utf-8"))
    client = TestClient(app)
    assert client.post("/api/scenario", json=data).status_code == 200
    raw = client.get("/api/dispatch/alt")
    assert raw.status_code == 200
    rows = [json.loads(line) for line in raw.text.splitlines() if line.strip()]
    alts = [row for row in rows if row["type"] == "alt"]
    assert [row["step"] for row in alts] == list(range(data["time"]["steps"] + 1))
    assert all("metrics" in row and "revenue_usd" in row["metrics"] for row in alts)
    assert alts[0]["metrics"]["revenue_usd"] == 0
    assert alts[-1]["metrics"]["jobs_completed"] >= 0


def _core(frame: dict) -> dict:
    return {key: frame[key] for key in ("step", "sat", "job", "missed", "metrics")}


def test_resume_add_jobs_replays_prefix(monkeypatch):
    """add_jobs на шаге 5: прошлое совпадает, сеть/greedy только с k, хвост расходится."""
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from orbi.greedy import greedy_action

    import console.api.main as main_mod
    from console.api import dispatch as dispatch_mod

    calls = {"n": 0}

    def actor_for(scenario):
        limit = int(scenario["model"]["downlink_parallel_limit"])

        def actor(obs, _limit=limit):
            calls["n"] += 1
            return greedy_action(obs, downlink_limit=_limit)

        return actor

    def actor_alt(scenario):
        limit = int(scenario["model"]["downlink_parallel_limit"])

        def actor(obs, _limit=limit):
            return greedy_action(obs, downlink_limit=_limit)

        return actor

    monkeypatch.setattr(dispatch_mod, "actor_for", actor_for)
    monkeypatch.setattr(dispatch_mod, "actor_for_alt", actor_alt)
    dispatch_mod.clear()
    main_mod._shift = None
    data = json.loads(_P01.read_text(encoding="utf-8"))
    event = {
        "id": "E-T",
        "at_step": 5,
        "type": "add_jobs",
        "jobs": [
            {
                "id": "URG-T",
                "kind": "relay",
                "release_step": 5,
                "deadline_step": 16,
                "work_steps": 3,
                "eligible_satellites": [sat["id"] for sat in data["satellites"]],
                "priority": 3,
                "value_usd": 999.0,
            }
        ],
    }

    run_a = dispatch_mod.Run("a")
    seq_a = list(
        dispatch_mod.iter_events(data, actor_for(data), run=run_a, goal="priority")
    )
    frames_a = [row for row in seq_a if row["type"] == "frame"]
    n_first = calls["n"]
    assert n_first == data["time"]["steps"]
    replay = [item.copy() for item in run_a.actions[:5]]
    calls["n"] = 0

    run_b = dispatch_mod.Run("b")
    seq_b = list(
        dispatch_mod.iter_events(
            data,
            actor_for(data),
            goal="revenue",
            scripted_events=[event],
            replay_actions=replay,
            from_step=5,
            run=run_b,
        )
    )
    frames_b = [row for row in seq_b if row["type"] == "frame"]
    assert calls["n"] == data["time"]["steps"] - 5
    assert frames_b[0]["step"] == 5
    assert _core(frames_a[5]) == _core(frames_b[0])
    assert any(job["id"] == "URG-T" for row in seq_b if row["type"] == "meta" for job in row["jobs"])
    assert any(
        _core(frames_a[row["step"]]) != _core(row)
        for row in frames_b
        if row["step"] > 5
    )

    client = TestClient(app)
    dispatch_mod.clear()
    main_mod._shift = None
    calls["n"] = 0
    assert client.post("/api/scenario", json=data).status_code == 200
    first = client.get("/api/dispatch")
    assert first.status_code == 200
    first_frames = [
        json.loads(line) for line in first.text.splitlines() if line.strip()
    ]
    first_frames = [row for row in first_frames if row["type"] == "frame"]
    n_http = calls["n"]
    calls["n"] = 0
    resumed = client.post(
        "/api/dispatch/resume",
        json={"step": 5, "events": [event], "goal": "priority"},
    )
    assert resumed.status_code == 200
    rows = [json.loads(line) for line in resumed.text.splitlines() if line.strip()]
    assert calls["n"] == n_http - 5
    resume_frames = [row for row in rows if row["type"] == "frame"]
    assert resume_frames[0]["step"] == 5
    assert _core(first_frames[5]) == _core(resume_frames[0])
    assert all(row["step"] >= 5 for row in resume_frames)
    assert any(job["id"] == "URG-T" for row in rows if row["type"] == "meta" for job in row["jobs"])


def test_openapi_covers_api_routes():
    client = TestClient(app)
    spec = client.get("/api/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    for route in (
        "/api/scenario",
        "/api/orbits",
        "/api/tracks",
        "/api/dispatch",
        "/api/dispatch/resume",
        "/api/dispatch/alt",
        "/api/dispatch/whatif",
        "/api/result",
    ):
        assert route in paths
    frame = paths["/api/dispatch"]["get"]
    assert "ckpt_68120" in spec.json()["info"]["description"]
    assert "ckpt_35714" in spec.json()["info"]["description"]
    yaml_body = client.get("/api/openapi.yaml")
    assert yaml_body.status_code == 200
    docs = client.get("/api/docs")
    assert docs.status_code == 200
    assert "whatif" in spec.json()["paths"]["/api/dispatch/whatif"]["post"]["operationId"].lower()


def test_whatif_replays_prefix_then_other_actor(monkeypatch):
    """whatif с шага 5: сеть второй политики только с k, метрики k как у primary."""
    import os

    import numpy as np

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from orbi.greedy import greedy_action

    import console.api.main as main_mod
    from console.api import dispatch as dispatch_mod

    primary = {"n": 0}
    other = {"n": 0}

    def actor_for(scenario):
        limit = int(scenario["model"]["downlink_parallel_limit"])

        def actor(obs, _limit=limit):
            primary["n"] += 1
            return greedy_action(obs, downlink_limit=_limit)

        return actor

    def actor_alt(scenario):
        def actor(obs):
            other["n"] += 1
            mask = np.asarray(obs["mask"], dtype=bool)
            n, width = mask.shape
            return np.full(n, width - 2, dtype=np.int64)

        return actor

    monkeypatch.setattr(dispatch_mod, "actor_for", actor_for)
    monkeypatch.setattr(dispatch_mod, "actor_for_alt", actor_alt)
    dispatch_mod.clear()
    main_mod._shift = None
    data = json.loads(_P01.read_text(encoding="utf-8"))
    client = TestClient(app)
    missing = client.post("/api/dispatch/whatif", json={"step": 5, "goal": "revenue"})
    assert missing.status_code == 409

    assert client.post("/api/scenario", json=data).status_code == 200
    first = client.get("/api/dispatch")
    assert first.status_code == 200
    primary_rows = [json.loads(line) for line in first.text.splitlines() if line.strip()]
    frames = [row for row in primary_rows if row["type"] == "frame"]
    n_primary = primary["n"]
    other["n"] = 0

    forked = client.post(
        "/api/dispatch/whatif",
        json={"step": 5, "goal": "revenue", "horizon": 10, "events": []},
    )
    assert forked.status_code == 200
    rows = [json.loads(line) for line in forked.text.splitlines() if line.strip()]
    meta = next(row for row in rows if row["type"] == "whatif_meta")
    what = [row for row in rows if row["type"] == "whatif"]
    assert meta["from_step"] == 5
    assert meta["horizon"] == 10
    assert meta["stay_step"] == 15
    assert what[0]["step"] == 5
    assert what[-1]["step"] == 15
    assert primary["n"] == n_primary
    assert other["n"] == 11
    assert what[0]["metrics"] == frames[5]["metrics"]
    assert what[-1]["metrics"] != frames[15]["metrics"]

    again = client.post(
        "/api/dispatch/whatif",
        json={"step": 5, "goal": "revenue", "horizon": 10, "events": []},
    )
    assert again.text == forked.text
    assert other["n"] == 11

    second = client.get("/api/dispatch")
    assert second.text == first.text

