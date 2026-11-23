"""Socket-blocked proofs for the offline paper path."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "trading" / "offline_paper.py"


def _load():
    spec = importlib.util.spec_from_file_location("offline_paper_under_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def offline():
    return _load()


@pytest.fixture
def blocked_sockets(monkeypatch):
    attempts: list[str] = []

    def _reject(*_args, **_kwargs):
        attempts.append("socket")
        raise OSError("socket blocked by offline test")

    monkeypatch.setattr(socket.socket, "connect", _reject)
    monkeypatch.setattr(socket, "create_connection", _reject)
    return attempts


def _install_fake_broker(calls: list[str]) -> None:
    def _record(name: str):
        def _call(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(name)

        return _call

    sys.modules["MetaTrader5"] = MagicMock(
        initialize=_record("initialize"),
        login=_record("login"),
        order_send=_record("order_send"),
        order_check=_record("order_check"),
        shutdown=_record("shutdown"),
    )
    sys.modules["metaapi_cloud_sdk"] = MagicMock(MetaApi=_record("MetaApi"))


def test_engine_starts_without_broker_and_places_no_external_orders(
    offline, blocked_sockets, tmp_path, capsys
):
    calls: list[str] = []
    _install_fake_broker(calls)
    ledger = tmp_path / "ledger.json"
    try:
        code = offline.run_offline_from_argv(
            [
                "--mode",
                "offline",
                "--bars",
                "8",
                "--symbol",
                "XAUUSD",
                "--algo",
                "ensemble",
                "--model-dir",
                str(tmp_path / "models"),
                "--ledger",
                str(ledger),
            ]
        )
    finally:
        sys.modules.pop("MetaTrader5", None)
        sys.modules.pop("metaapi_cloud_sdk", None)

    captured = capsys.readouterr()
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["engine_started"] is True
    assert payload["external_order_count"] == 0
    assert payload["local_fill_count"] == 0
    assert payload["bar_source"] == "local-synthetic"
    assert "missing_trained_weights" in payload["halt_reasons"]
    assert blocked_sockets == []
    assert calls == []
    assert "external_order_count=0" in captured.out


def test_broker_secrets_are_not_leaked_or_used(
    offline, blocked_sockets, tmp_path, monkeypatch, capsys
):
    secret = "offline-proof-secret-value"
    token = "offline-proof-metaapi-token"
    monkeypatch.setenv("MT5_PASSWORD", secret)
    monkeypatch.setenv("METAAPI_TOKEN", token)
    monkeypatch.setenv("MT5_SERVER", "Broker-Demo-Should-Not-Appear")
    ledger = tmp_path / "ledger.json"
    calls: list[str] = []
    _install_fake_broker(calls)
    try:
        code = offline.run_offline_from_argv(
            [
                "--mode",
                "offline",
                "--bars",
                "4",
                "--ledger",
                str(ledger),
                "--model-dir",
                str(tmp_path),
            ]
        )
    finally:
        sys.modules.pop("MetaTrader5", None)
        sys.modules.pop("metaapi_cloud_sdk", None)

    captured = capsys.readouterr()
    text = ledger.read_text(encoding="utf-8") + captured.out + captured.err
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert code == 0
    assert payload["external_order_count"] == 0
    assert secret not in text
    assert token not in text
    assert "Broker-Demo-Should-Not-Appear" not in text
    assert "MT5_PASSWORD" in payload["ignored_broker_env"]
    assert "METAAPI_TOKEN" in payload["ignored_broker_env"]
    assert calls == []
    assert blocked_sockets == []


def test_broker_modes_stay_on_their_own_path(monkeypatch):
    from main import _offline_cli_requested

    monkeypatch.delenv("MODE", raising=False)
    assert _offline_cli_requested(["--mode", "offline"]) is True
    assert _offline_cli_requested(["--mode", "demo"]) is False
    assert _offline_cli_requested(["--mode", "live"]) is False
    assert _offline_cli_requested(["--mode", "backtest"]) is False
    assert _offline_cli_requested(["--doctor"]) is False
    assert _offline_cli_requested([]) is False


def test_live_flag_is_rejected(offline, capsys):
    assert offline.run_offline_from_argv(["--mode", "live"]) == 2
    assert offline.run_offline_from_argv(["--mode", "offline", "--confirm-live"]) == 2
    captured = capsys.readouterr()
    assert "Refusing live trading" in captured.out


def test_local_fill_does_not_call_broker(offline, blocked_sockets):
    calls: list[str] = []
    _install_fake_broker(calls)
    try:
        broker = offline.OfflinePaperBroker(Path("unused.json"))
        broker.connect()
        assert broker.supervisor.trading_allowed
        ticket = broker.place_order(
            offline.PaperOrder(symbol="XAUUSD", direction=1, lot_size=0.01, price=2300.0)
        )
        assert ticket == 1
        assert broker.external_order_count == 0
        assert broker.positions[0]["ticket"] == 1
        with pytest.raises(offline.OfflineSafetyError):
            sys.modules["MetaTrader5"].order_send({"symbol": "XAUUSD"})
        with pytest.raises(offline.OfflineSafetyError):
            sys.modules["MetaTrader5"].initialize()
    finally:
        sys.modules.pop("MetaTrader5", None)
        sys.modules.pop("metaapi_cloud_sdk", None)
    assert calls == []
    assert blocked_sockets == []


def test_missing_weights_block_place_order(offline, tmp_path):
    payload = offline.run_offline_session(
        symbol="XAUUSD",
        algo="lstm",
        bars=3,
        seed=1,
        model_dir=tmp_path,
        ledger_path=tmp_path / "ledger.json",
    )
    assert payload["external_order_count"] == 0
    assert payload["local_fill_count"] == 0
    assert payload["trading_allowed"] is False
    broker = offline.OfflinePaperBroker(tmp_path / "other.json")
    broker.connect()
    broker.supervisor.halted = True
    broker.supervisor.halt_reasons = ("missing_trained_weights",)
    with pytest.raises(offline.OfflineOrderBlocked):
        broker.place_order(offline.PaperOrder("XAUUSD", 1, 0.01, 2300.0))


def test_main_entrypoint_with_secrets_in_the_shell(tmp_path):
    secret = "entrypoint-secret-must-not-leak"
    env = os.environ.copy()
    env["MT5_PASSWORD"] = secret
    env["METAAPI_TOKEN"] = "entrypoint-metaapi-secret"
    env.pop("MODE", None)
    ledger = tmp_path / "ledger.json"
    proc = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--mode",
            "offline",
            "--bars",
            "5",
            "--ledger",
            str(ledger),
            "--model-dir",
            str(tmp_path / "models"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    combined = proc.stdout + proc.stderr + ledger.read_text(encoding="utf-8")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["engine_started"] is True
    assert payload["external_order_count"] == 0
    assert secret not in combined
    assert "entrypoint-metaapi-secret" not in combined


def test_source_has_no_broker_order_calls():
    tree_text = MODULE_PATH.read_text(encoding="utf-8")
    assert "mt5.order_send" not in tree_text
    assert "create_market_order" not in tree_text
    assert "import MetaTrader5" not in tree_text
    assert "from metaapi_cloud_sdk import" not in tree_text
