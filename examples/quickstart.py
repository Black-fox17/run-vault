"""RunVault Quickstart Example.

Demonstrates local experiment tracking, parameter logging, metric recording,
and querying previous runs using the RunVault Python SDK.
"""

import math
import random

import runvault


def simulate_training() -> None:
    print("=" * 60)
    print(" RunVault — Local Experiment Tracking Quickstart")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Start a Training Run (Trial 1: Baseline)
    # ---------------------------------------------------------
    print("\n[1] Starting Experiment: 'cifar10-convnet' -> Run: 'baseline-sgd'")
    with runvault.init(
        experiment="cifar10-convnet",
        name="baseline-sgd",
        tags={"architecture": "simple-cnn", "optimizer": "sgd"},
        description="Convolutional net benchmark on CIFAR-10",
    ) as run:
        print(f"    Active Run ID: {run.id}")
        print(f"    Status: {run.status.value}")

        # Log hyperparameters
        run.log_params(
            {
                "learning_rate": 0.01,
                "batch_size": 64,
                "momentum": 0.9,
                "weight_decay": 1e-4,
                "layers": [32, 64, 128],
            }
        )
        print("    Logged configuration parameters.")

        # Simulate 10 training epochs
        print("    Streaming metrics across training epochs...")
        for epoch in range(10):
            loss = 1.8 / (1.0 + 0.3 * epoch) + random.uniform(-0.02, 0.02)
            acc = 0.45 + 0.45 * (1.0 - math.exp(-0.25 * epoch)) + random.uniform(-0.01, 0.01)

            run.log_metrics(
                {
                    "train_loss": round(loss, 4),
                    "val_loss": round(loss + 0.05, 4),
                    "accuracy": round(acc, 4),
                },
                step=epoch,
            )

        run.set_tags({"benchmark": "passed"})

    print(f"    Run complete. Final status: {run.status.value}")

    # ---------------------------------------------------------
    # 2. Start a Second Run (Trial 2: AdamW with higher LR)
    # ---------------------------------------------------------
    print("\n[2] Starting Second Run: 'tuned-adamw'")
    with runvault.init(
        experiment="cifar10-convnet",
        name="tuned-adamw",
        tags={"architecture": "simple-cnn", "optimizer": "adamw"},
    ) as run2:
        run2.log_params(
            {
                "learning_rate": 0.001,
                "batch_size": 128,
                "weight_decay": 1e-2,
                "layers": [32, 64, 128],
            }
        )

        for epoch in range(10):
            loss = 1.5 / (1.0 + 0.45 * epoch) + random.uniform(-0.01, 0.01)
            acc = 0.52 + 0.42 * (1.0 - math.exp(-0.35 * epoch)) + random.uniform(-0.01, 0.01)

            run2.log_metrics(
                {
                    "train_loss": round(loss, 4),
                    "val_loss": round(loss + 0.03, 4),
                    "accuracy": round(acc, 4),
                },
                step=epoch,
            )

    print(f"    Run complete. Final status: {run2.status.value}")

    # ---------------------------------------------------------
    # 3. Query and Inspect Previous Runs
    # ---------------------------------------------------------
    print("\n[3] Querying Persisted Experiment Metadata from SQLite:")
    exp = runvault.get_experiment_by_name("cifar10-convnet")
    if exp:
        print(f"    Experiment: {exp.name} (ID: {exp.id})")
        print(f"    Description: {exp.description}")

        runs = exp.list_runs()
        print(f"\n    Found {len(runs)} runs:")
        print(f"    {'Run Name':<16} | {'Status':<10} | {'Duration':<10} | {'Params'}")
        print("    " + "-" * 65)

        for r in runs:
            params = runvault.get_params(r.id)
            lr = params.get("learning_rate")
            bs = params.get("batch_size")
            dur = f"{r.duration_seconds:.2f}s" if r.duration_seconds is not None else "N/A"
            name_col = r.name or "unnamed"
            print(f"    {name_col:<16} | {r.status.value:<10} | {dur:<10} | lr={lr}, bs={bs}")

        # Inspect metric history of the best run
        print(f"\n[4] Metric History ('accuracy') for '{run2.name}':")
        history = runvault.get_metric_history(run2.id, metric_name="accuracy")
        for point in history:
            ts_str = point.timestamp.strftime("%H:%M:%S")
            print(f"    Epoch {point.step:>2}: accuracy = {point.value:.4f} (at {ts_str})")

    print("\n" + "=" * 60)
    print(" Quickstart test completed successfully!")
    print(" Local SQLite database stored at: .runvault/metadata.db")
    print("=" * 60)


if __name__ == "__main__":
    simulate_training()
