import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from aggregation import run_federated
from client_training import train_local_update
from constants import DEFAULT_MODEL_DIR
from scoring import score_csv
from storage import global_model_path, initialize_global_model, load_global_model, read_json
from validation import clean_feature_names


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TrustNet federated learning server and client utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create a zero-initialized global model.")
    init_parser.add_argument("--feature-columns", required=True, help="Comma-separated feature list.")
    init_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    init_parser.add_argument("--created-by", default="trustnet")

    train_parser = subparsers.add_parser(
        "train-client",
        help="Train a local update from a real CSV without exporting raw rows.",
    )
    train_parser.add_argument("--csv", required=True, dest="csv_path")
    train_parser.add_argument("--target-column", required=True)
    train_parser.add_argument("--client-id", required=True)
    train_parser.add_argument("--round-id", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--feature-columns", help="Comma-separated feature list.")
    train_parser.add_argument("--global-model", help="Path to global_model.json or its model dir.")
    train_parser.add_argument("--epochs", type=int, default=200)
    train_parser.add_argument("--learning-rate", type=float, default=0.01)
    train_parser.add_argument("--l2", type=float, default=0.001)
    train_parser.add_argument("--hmac-secret")

    aggregate_parser = subparsers.add_parser(
        "aggregate",
        help="Aggregate one federated round from client update JSON files.",
    )
    aggregate_parser.add_argument("--round-id", required=True)
    aggregate_parser.add_argument("--updates", nargs="+", required=True)
    aggregate_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    aggregate_parser.add_argument("--min-clients", type=int, default=2)
    aggregate_parser.add_argument("--min-samples-per-client", type=int, default=1)
    aggregate_parser.add_argument("--max-update-norm", type=float, default=25.0)
    aggregate_parser.add_argument("--server-learning-rate", type=float, default=1.0)
    aggregate_parser.add_argument("--require-existing-model", action="store_true")
    aggregate_parser.add_argument("--hmac-secret")

    predict_parser = subparsers.add_parser("predict", help="Score a CSV with the global model.")
    predict_parser.add_argument("--input-csv", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))

    show_parser = subparsers.add_parser("show-model", help="Print global model metadata.")
    show_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))

    return parser


def handle_cli(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "init":
        model = initialize_global_model(
            feature_columns=clean_feature_names(args.feature_columns),
            model_dir=args.model_dir,
            created_by=args.created_by,
        )
        return {
            "status": "success",
            "model_version": model["model_version"],
            "model_hash": model["model_hash"],
            "schema_hash": model["schema_hash"],
            "global_model_path": str(global_model_path(Path(args.model_dir))),
        }

    if args.command == "train-client":
        update = train_local_update(
            csv_path=args.csv_path,
            target_column=args.target_column,
            client_id=args.client_id,
            round_id=args.round_id,
            output_path=args.output,
            feature_columns=(
                clean_feature_names(args.feature_columns) if args.feature_columns else None
            ),
            global_model_path=args.global_model,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            hmac_secret=args.hmac_secret,
        )
        return {
            "status": "success",
            "client_id": update["client_id"],
            "round_id": update["round_id"],
            "sample_count": update["sample_count"],
            "schema_hash": update["schema_hash"],
            "output": args.output,
            "metrics": update["metrics"],
        }

    if args.command == "aggregate":
        return run_federated(
            {
                "round_id": args.round_id,
                "client_updates": read_updates(args.updates),
                "config": {
                    "model_dir": args.model_dir,
                    "min_clients": args.min_clients,
                    "min_samples_per_client": args.min_samples_per_client,
                    "max_update_norm": args.max_update_norm,
                    "server_learning_rate": args.server_learning_rate,
                    "require_existing_model": args.require_existing_model,
                    "hmac_secret": args.hmac_secret,
                },
            }
        )

    if args.command == "predict":
        result = score_csv(args.input_csv, args.output, args.model_dir)
        return {
            "status": "success",
            "model_version": result["model_version"],
            "predictions": len(result["predictions"]),
            "output": args.output,
        }

    if args.command == "show-model":
        model = load_global_model(Path(args.model_dir), required=True)
        return {
            "status": "success",
            "model_id": model["model_id"],
            "model_version": model["model_version"],
            "schema_hash": model["schema_hash"],
            "feature_names": model["feature_names"],
            "rounds_completed": model.get("rounds_completed", 0),
            "last_round_id": model.get("last_round_id"),
            "updated_at": model.get("updated_at"),
        }

    raise ValueError(f"Unknown command: {args.command}")


def read_updates(paths: Sequence[str]) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for path in paths:
        loaded = read_json(Path(path))
        if isinstance(loaded, list):
            updates.extend(loaded)
        else:
            updates.append(loaded)
    return updates


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()
    try:
        result = handle_cli(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
