"""Command line interface: generate data, train, evaluate, check drift, serve."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from driftsentry.config import DataConfig, DriftSentryConfig


def _load_transactions(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, parse_dates=["timestamp"])


def cmd_generate(args: argparse.Namespace) -> None:
    from driftsentry.data.synthetic import generate_transactions

    cfg = DataConfig(n_accounts=args.accounts, days=args.days, seed=args.seed)
    df = generate_transactions(
        cfg, amount_shift=args.amount_shift, anomalies=not args.no_anomalies
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(
        f"wrote {len(df)} transactions for {df['account_id'].nunique()} accounts "
        f"({int(df['is_anomaly'].sum())} anomalous) to {out}"
    )


def cmd_train(args: argparse.Namespace) -> None:
    from driftsentry.pipeline import run_training

    cfg = DriftSentryConfig()
    cfg.train.epochs = args.epochs
    cfg.train.target_recall = args.target_recall
    cfg.train.device = args.device
    cfg.window.seq_len = args.seq_len
    if args.data:
        df = _load_transactions(args.data)
    else:
        from driftsentry.data.synthetic import generate_transactions

        df = generate_transactions(cfg.data)
    bundle = run_training(df, cfg, output_dir=args.out)
    print(json.dumps(bundle.metrics, indent=2, default=float))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from driftsentry.artifacts import load_bundle
    from driftsentry.pipeline import evaluate_bundle

    bundle = load_bundle(args.artifacts)
    report = evaluate_bundle(bundle, _load_transactions(args.data))
    print(json.dumps(report, indent=2, default=float))


def cmd_drift(args: argparse.Namespace) -> None:
    from driftsentry.api.service import InferenceService
    from driftsentry.artifacts import load_bundle

    service = InferenceService(load_bundle(args.artifacts))
    df = _load_transactions(args.data)
    cols = ["txn_id", "account_id", "timestamp", "amount", "txn_type", "channel"]
    report = service.drift(df[[c for c in cols if c in df.columns]].to_dict("records"))
    print(json.dumps(report, indent=2, default=float))


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ.setdefault("DRIFTSENTRY_ARTIFACTS_DIR", args.artifacts)
    uvicorn.run("driftsentry.api.main:app", host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="driftsentry", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate synthetic transactions")
    g.add_argument("--out", default="data/transactions.csv")
    g.add_argument("--accounts", type=int, default=400)
    g.add_argument("--days", type=int, default=120)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--amount-shift", type=float, default=0.0,
                   help="shift in log-amount mean, simulates population drift")
    g.add_argument("--no-anomalies", action="store_true")
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="train the autoencoder and save a model bundle")
    t.add_argument("--data", help="CSV/parquet of transactions; synthetic if omitted")
    t.add_argument("--out", default="artifacts")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--seq-len", type=int, default=20)
    t.add_argument("--target-recall", type=float, default=0.94)
    t.add_argument("--device", default="cpu")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="evaluate a bundle on labeled transactions")
    e.add_argument("--artifacts", default="artifacts")
    e.add_argument("--data", required=True)
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("drift", help="run a drift check on a batch of transactions")
    d.add_argument("--artifacts", default="artifacts")
    d.add_argument("--data", required=True)
    d.set_defaults(func=cmd_drift)

    s = sub.add_parser("serve", help="start the FastAPI inference service")
    s.add_argument("--artifacts", default="artifacts")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
