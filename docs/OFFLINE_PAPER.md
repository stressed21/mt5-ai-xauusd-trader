# Offline paper mode

`python main.py --mode offline` runs a local paper session. It does not log in to a broker, and it does not send an order.

```bash
python main.py --mode offline --symbol XAUUSD
```

Useful flags:

```bash
python main.py --mode offline --symbol XAUUSD --bars 64 --seed 42 \
  --model-dir models/trained --ledger data/offline/ledger.json
```

The process writes `data/offline/ledger.json`. A completed run reports `external_order_count=0`. Local bars are a seeded synthetic series generated in memory. The session records fills only on that ledger, and only when the safety supervisor allows it.

## What this mode refuses

- MetaTrader `initialize`, `login`, and `order_send`.
- MetaAPI `MetaApi` and `create_market_order`.
- Any other broker account or order call. If those libraries are already imported, their entry points are replaced with functions that raise.
- `--mode live` and `--confirm-live` on this path.
- Broker values in `MT5_PASSWORD`, `MT5_LOGIN`, `MT5_SERVER`, `MT5_PATH`, `METAAPI_TOKEN`, `METAAPI_ACCOUNT_ID`, `TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`, `DATABASE_URL`, and `REDIS_URL`. The names may be listed as ignored. The values are not read into the ledger or the log.

The safety supervisor is attached for every offline session. It starts in the halted state for this build, so the default run does not open a local fill.

## Missing trained weights

The repository does not contain `models/trained/ppo_xauusd.zip` or `models/trained/lstm_xauusd.pt`. Ensemble and LSTM sessions halt with `missing_trained_weights` and leave `local_fill_count` at 0.

Do not drop in an empty file to bypass that check. The stock LSTM loader builds a network with random weights when a checkpoint is missing or cannot be applied. This offline path does not call that loader. If the weight files are present, the session still halts with `model_loader_refuses_random_weights` rather than execute a model that can fall back to a random initialization.

On the broker paths, `--algo lstm` now returns before constructing that random network when `lstm_xauusd.pt` is absent. `MODE=demo` and `MODE=live` are otherwise unchanged: both still call the broker order path once a session exists. Broker demo mode is not this offline mode.

## Limits that remain

- The native `MetaTrader5` wheel is installed only on Windows (`sys_platform == 'win32'`). It is not used by offline mode.
- `metaapi-cloud-sdk` is not pinned in the requirements files. A Mac or Linux broker session is not available from the pinned dependencies, and the current `python-socketio==5.18.1` pin does not match the older socketio range described in the audit notes.
- Backtest still calls `MT5Connector.connect()` before it reads history. Use `--mode offline` when the machine must stay off the broker.
- Docker Compose defaults to `MODE=offline`. The image command is `python main.py --mode offline`.
- This mode does not report strategy performance. The README performance table lists targets. The walk-forward audit file does not publish those figures.
- No broker credentials belong in this repository.
