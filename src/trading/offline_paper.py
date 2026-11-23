"""Fail-closed offline paper trading.

This module is loaded without importing the broker connector. It uses a local
bar generator and a local ledger. It does not call MetaTrader, MetaAPI, or any
account or order API.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

BROKER_ENV_NAMES = (
    "MT5_LOGIN",
    "MT5_PASSWORD",
    "MT5_SERVER",
    "MT5_PATH",
    "METAAPI_TOKEN",
    "METAAPI_ACCOUNT_ID",
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
    "DATABASE_URL",
    "REDIS_URL",
)

_WEIGHT_FILES = {
    "ensemble": ("ppo_xauusd.zip", "lstm_xauusd.pt"),
    "ppo": ("ppo_xauusd.zip",),
    "lstm": ("lstm_xauusd.pt",),
    "dreamer": ("dreamer_xauusd.pt",),
    "transformer": ("transformer_xauusd.pt",),
}

_FORBIDDEN_BROKER_ATTRS = (
    "initialize",
    "login",
    "order_send",
    "order_check",
    "shutdown",
)


class OfflineSafetyError(RuntimeError):
    """Raised when offline mode is asked to touch a broker API."""


class OfflineOrderBlocked(OfflineSafetyError):
    """Raised when the safety supervisor refuses a local fill."""


@dataclass(frozen=True)
class PaperOrder:
    """Local order request. This object never leaves the process."""

    symbol: str
    direction: int
    lot_size: float
    price: float


def _load_safety_supervisor() -> Any:
    """Load SafetySupervisor without importing src/trading/__init__.py."""

    path = Path(__file__).resolve().with_name("safety_supervisor.py")
    spec = importlib.util.spec_from_file_location("offline_safety_supervisor", path)
    if spec is None or spec.loader is None:
        raise OfflineSafetyError("Safety supervisor module is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.SafetySupervisor


def arm_broker_tripwires() -> None:
    """Replace already-imported broker entry points so they cannot be called."""

    def _blocked(name: str) -> Any:
        def _call(*_args: Any, **_kwargs: Any) -> None:
            raise OfflineSafetyError(f"offline mode blocked broker call: {name}")

        return _call

    mt5 = sys.modules.get("MetaTrader5")
    if mt5 is not None:
        for attr in _FORBIDDEN_BROKER_ATTRS:
            setattr(mt5, attr, _blocked(f"MetaTrader5.{attr}"))

    meta = sys.modules.get("metaapi_cloud_sdk")
    if meta is not None:
        meta.MetaApi = _blocked("metaapi_cloud_sdk.MetaApi")


def present_broker_env_names() -> list[str]:
    """Return names of broker variables that are set. Values are not read."""

    names: list[str] = []
    for name in BROKER_ENV_NAMES:
        if os.environ.get(name):
            names.append(name)
    return names


def required_weight_files(algo: str, model_dir: Path) -> list[Path]:
    names = _WEIGHT_FILES.get(algo)
    if names is None:
        return [model_dir / f"{algo}_weights_unspecified"]
    return [model_dir / name for name in names]


def missing_weight_files(algo: str, model_dir: Path) -> list[str]:
    return [str(path) for path in required_weight_files(algo, model_dir) if not path.is_file()]


def generate_local_bars(symbol: str, count: int, seed: int) -> list[dict[str, Any]]:
    """Build a deterministic OHLCV path in memory. No file download and no socket."""

    if count < 1:
        raise OfflineSafetyError("bar count must be positive")
    rng = random.Random(seed)
    price = 2300.0
    start = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
    bars: list[dict[str, Any]] = []
    for step in range(count):
        drift = rng.uniform(-1.25, 1.25)
        close = round(max(1.0, price + drift), 2)
        high = round(max(price, close) + rng.uniform(0.0, 0.4), 2)
        low = round(min(price, close) - rng.uniform(0.0, 0.4), 2)
        bars.append(
            {
                "symbol": symbol,
                "time": (start + timedelta(minutes=5 * step)).isoformat(),
                "open": price,
                "high": high,
                "low": low,
                "close": close,
                "tick_volume": 100 + step,
                "source": "local-synthetic",
            }
        )
        price = close
    return bars


class OfflinePaperBroker:
    """Local account, positions, and fills. External order count stays zero."""

    def __init__(self, ledger_path: Path, supervisor: Any | None = None) -> None:
        self.ledger_path = ledger_path
        self.supervisor = supervisor if supervisor is not None else _load_safety_supervisor()()
        self.cash = 10000.0
        self.equity = 10000.0
        self.positions: list[dict[str, Any]] = []
        self.fills: list[dict[str, Any]] = []
        self.external_order_count = 0
        self._next_ticket = 1
        self._initialized = False
        self._last_price = 2300.0

    def connect(self) -> bool:
        """Mark the local session ready. This does not open a broker connection."""

        arm_broker_tripwires()
        self._initialized = True
        self.reconcile()
        return True

    def disconnect(self) -> None:
        self._initialized = False

    def reconcile(self) -> Any:
        return self.supervisor.reconcile(self, list(self.positions))

    def get_account_info(self) -> dict[str, Any]:
        return {
            "balance": self.cash,
            "equity": self.equity,
            "trade_allowed": False,
        }

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        if symbol is None:
            return list(self.positions)
        return [row for row in self.positions if row.get("symbol") == symbol]

    def get_terminal_status(self) -> dict[str, Any]:
        return {"algo_trading": True, "offline": True, "trade_allowed": False}

    def mark(self, price: float) -> None:
        self._last_price = price
        self.equity = self.cash

    def place_order(self, order: PaperOrder) -> int:
        """Record a local fill. Broker order functions are not called."""

        if not self._initialized:
            raise OfflineSafetyError("offline session is not started")
        if not self.supervisor.trading_allowed:
            raise OfflineOrderBlocked("safety supervisor is not allowing local fills")
        self.reconcile()
        if not self.supervisor.trading_allowed:
            raise OfflineOrderBlocked("safety supervisor halted during reconciliation")

        ticket = self._next_ticket
        self._next_ticket += 1
        fill = {
            "ticket": ticket,
            "symbol": order.symbol,
            "direction": order.direction,
            "lot_size": order.lot_size,
            "price": order.price,
            "venue": "local-ledger",
            "external": False,
        }
        self.fills.append(fill)
        self.positions.append(
            {
                "ticket": ticket,
                "symbol": order.symbol,
                "volume": order.lot_size,
                "type": order.direction,
                "price_open": order.price,
            }
        )
        self.reconcile()
        return ticket

    def snapshot(self, **extra: Any) -> dict[str, Any]:
        payload = {
            "mode": "offline",
            "engine_started": self._initialized,
            "external_order_count": self.external_order_count,
            "local_fill_count": len(self.fills),
            "fills": self.fills,
            "positions": self.positions,
            "halt_reasons": list(self.supervisor.halt_reasons),
            "trading_allowed": self.supervisor.trading_allowed,
            "cash": self.cash,
            "equity": self.equity,
            "last_price": self._last_price,
        }
        payload.update(extra)
        return payload

    def write_ledger(self, **extra: Any) -> dict[str, Any]:
        payload = self.snapshot(**extra)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(self.ledger_path, 0o600)
        return payload


def run_offline_session(
    *,
    symbol: str,
    algo: str,
    bars: int,
    seed: int,
    model_dir: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    """Start the offline engine, walk local bars, and write the ledger."""

    arm_broker_tripwires()
    broker = OfflinePaperBroker(ledger_path)
    broker.connect()
    local_bars = generate_local_bars(symbol, bars, seed)
    missing = missing_weight_files(algo, model_dir)
    if missing:
        broker.supervisor.halted = True
        broker.supervisor.halt_reasons = ("missing_trained_weights",)
    else:
        # Stock model constructors can fall back to random weights when a load
        # fails. This path does not call them.
        broker.supervisor.halted = True
        broker.supervisor.halt_reasons = ("model_loader_refuses_random_weights",)

    for bar in local_bars:
        broker.mark(float(bar["close"]))
        if broker.supervisor.trading_allowed:
            raise OfflineSafetyError("offline session attempted a fill without a safe model")

    ignored = present_broker_env_names()
    return broker.write_ledger(
        symbol=symbol,
        algorithm=algo,
        bar_source="local-synthetic",
        bar_count=len(local_bars),
        missing_weights=missing,
        ignored_broker_env=ignored,
    )


def run_offline_from_argv(argv: list[str] | None = None) -> int:
    """CLI entry used by ``python main.py --mode offline``."""

    parser = argparse.ArgumentParser(
        prog="python main.py --mode offline",
        description="Run a local paper session. Broker credentials are ignored.",
    )
    parser.add_argument("--mode", default="offline")
    parser.add_argument("--confirm-live", action="store_true")
    parser.add_argument("--symbol", default="XAUUSD")
    parser.add_argument("--algo", default="ensemble")
    parser.add_argument("--bars", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=Path("models/trained"))
    parser.add_argument("--ledger", type=Path, default=Path("data/offline/ledger.json"))
    args, _unknown = parser.parse_known_args(argv)

    if args.confirm_live or args.mode == "live":
        print("Refusing live trading on the offline paper path.")
        return 2
    if args.mode not in {"offline", "paper"}:
        print("Refusing broker mode on the offline paper path.")
        return 2

    try:
        payload = run_offline_session(
            symbol=args.symbol,
            algo=args.algo,
            bars=args.bars,
            seed=args.seed,
            model_dir=args.model_dir,
            ledger_path=args.ledger,
        )
    except OfflineSafetyError as exc:
        print(f"Offline paper session failed closed: {exc}")
        return 3

    print("Offline paper engine started.")
    print(f"Ledger: {args.ledger}")
    print(f"Bars: {payload['bar_count']} source={payload['bar_source']}")
    print(f"external_order_count={payload['external_order_count']}")
    print(f"local_fill_count={payload['local_fill_count']}")
    if payload["halt_reasons"]:
        print("fail_closed=" + ",".join(payload["halt_reasons"]))
    if payload["ignored_broker_env"]:
        print("Ignored broker environment names: " + ", ".join(payload["ignored_broker_env"]))
    return 0
