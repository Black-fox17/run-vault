"""RunVault Quickstart Example.

Demonstrates local experiment tracking, parameter logging, metric recording,
atomic checkpoint publication, and fault-tolerant training recovery.
"""

import math
import random
from pathlib import Path

import runvault


def _run_baseline_trial() -> None:
    """Execute a baseline trial logging parameters and metrics."""
    print("\n[1] Starting Experiment: 'cifar10-convnet' -> Run: 'baseline-sgd'")
    with runvault.init(
        experiment="cifar10-convnet",
        name="baseline-sgd",
        tags={"architecture": "simple-cnn", "optimizer": "sgd"},
        description="Convolutional net benchmark on CIFAR-10",
    ) as run:
        print(f"    Active Run ID: {run.id}")
        print(f"    Status: {run.status.value}")

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

        print("    Streaming metrics across training epochs...")
        for epoch in range(5):
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


def _run_tuned_trial_with_checkpoints() -> str:
    """Execute a tuned trial that saves atomic checkpoints."""
    print("\n[2] Starting Second Run with Checkpoints: 'tuned-adamw'")
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

        for epoch in range(1, 7):
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

            # Persist atomic checkpoints at epoch 3 and epoch 6
            if epoch in (3, 6):

                def save_weights(dest: Path, ep: int = epoch, accuracy: float = acc) -> None:
                    # In real training: torch.save(model.state_dict(), dest)
                    dest.write_text(f"MODEL_WEIGHTS_EPOCH_{ep}_ACC_{accuracy:.4f}\n")

                ckpt = run2.save_checkpoint(
                    name=f"checkpoint-epoch-{epoch}",
                    writer=save_weights,
                    step=epoch,
                    metadata={"epoch": epoch, "accuracy": round(acc, 4), "loss": round(loss, 4)},
                )
                print(
                    f"    Saved Checkpoint: '{ckpt.name}' (Step {ckpt.step}, "
                    f"Commit #{ckpt.commit_seq}, SHA-256: {ckpt.checksum_sha256[:8]}...)"
                )

        print(f"    Run complete. Final status: {run2.status.value}")
        return run2.id


def _demonstrate_checkpoint_recovery(run_id: str) -> Path:
    """Demonstrate direct checkpoint recovery using PathLike recovery results."""
    print("\n[3] Restoring Checkpoint for Training Recovery:")
    recovery = runvault.restore_checkpoint(run_id)
    print(f"    Restored Checkpoint:  {recovery.checkpoint.name}")
    print(f"    Commit Sequence:      #{recovery.checkpoint.commit_seq}")
    print(f"    Resolved Path:        {recovery.path}")
    print(f"    Payload Content:      {recovery.read_text().strip()}")
    print(f"    Is Fallback Recovery: {recovery.is_fallback}")
    return recovery.path


def _demonstrate_fault_tolerant_fallback(run_id: str, latest_ckpt_file: Path) -> None:
    """Demonstrate automatic fallback when a checkpoint artifact is corrupt."""
    print("\n[4] Simulating Checkpoint Corruption & Automatic Fallback:")
    original_bytes = latest_ckpt_file.read_bytes()
    latest_ckpt_file.write_bytes(b"CORRUPTED_PARTIAL_BYTES")

    print(f"    Corrupted latest checkpoint file on disk: {latest_ckpt_file.name}")
    print("    Requesting latest valid checkpoint restore...")

    fallback_recovery = runvault.restore_checkpoint(run_id)
    print(f"    Fell back to Checkpoint: {fallback_recovery.checkpoint.name}")
    print(f"    Commit Sequence:         #{fallback_recovery.checkpoint.commit_seq}")
    print(f"    Payload Content:         {fallback_recovery.read_text().strip()}")
    print(f"    Is Fallback:             {fallback_recovery.is_fallback}")
    if fallback_recovery.rejected_checkpoints:
        print(f"    Rejected Candidates:     {len(fallback_recovery.rejected_checkpoints)}")
        for rejected_ckpt, reason in fallback_recovery.rejected_checkpoints:
            print(f"      - {rejected_ckpt.name}: {reason}")

    # Restore original bytes so workspace remains healthy
    latest_ckpt_file.write_bytes(original_bytes)


def _query_experiment_metadata(tuned_run_id: str) -> None:
    """Query experiments, runs, and checkpoints from local SQLite."""
    print("\n[5] Querying Persisted Experiment Metadata from SQLite:")
    exp = runvault.get_experiment_by_name("cifar10-convnet")
    if exp:
        print(f"    Experiment: {exp.name} (ID: {exp.id})")
        runs = exp.list_runs()
        print(f"\n    Found {len(runs)} runs:")
        print(f"    {'Run Name':<16} | {'Status':<10} | {'Duration':<10} | {'Checkpoints'}")
        print("    " + "-" * 65)

        for r in runs:
            ckpts = runvault.list_checkpoints(r.id)
            dur = f"{r.duration_seconds:.2f}s" if r.duration_seconds is not None else "N/A"
            name_col = r.name or "unnamed"
            print(f"    {name_col:<16} | {r.status.value:<10} | {dur:<10} | {len(ckpts)} saved")

        print(f"\n[6] Detailed Checkpoint Listing for Run '{tuned_run_id[:8]}...':")
        for c in runvault.list_checkpoints(tuned_run_id):
            print(
                f"    Commit #{c.commit_seq:<2} | {c.name:<22} | "
                f"Step: {c.step} | Size: {c.size_bytes}B | SHA: {c.checksum_sha256[:12]}..."
            )


def simulate_training() -> None:
    """Run full quickstart simulation showing experiment tracking and checkpointing."""
    print("=" * 70)
    print(" RunVault — Experiment Tracking & Checkpoint Recovery Quickstart")
    print("=" * 70)

    _run_baseline_trial()
    tuned_run_id = _run_tuned_trial_with_checkpoints()
    latest_ckpt_path = _demonstrate_checkpoint_recovery(tuned_run_id)
    _demonstrate_fault_tolerant_fallback(tuned_run_id, latest_ckpt_path)
    _query_experiment_metadata(tuned_run_id)

    print("\n" + "=" * 70)
    print(" Quickstart completed successfully!")
    print(" Local SQLite database: .runvault/metadata.db")
    print(" Checkpoints directory: .runvault/checkpoints/")
    print("=" * 70)


if __name__ == "__main__":
    simulate_training()
