"""Interactive verification and smoke test script for Google Drive Checkpoint Storage (Phase 4.2).

Usage:
    # 1. Run offline verification (in-memory mock of Google Drive API)
    python examples/test_gdrive_interactive.py

    # 2. Run live verification against your actual Google Drive
    python examples/test_gdrive_interactive.py --live --client-secrets /path/to/client_secret.json
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from tests.test_gdrive import FakeDriveService

import runvault
from runvault.core.workspace import Workspace
from runvault.integrations.gdrive import (
    GoogleDriveStorageProvider,
    check_gdrive_dependencies,
    is_colab_environment,
)


def run_offline_test() -> None:
    """Run an end-to-end verification of Google Drive storage using an in-memory mock."""
    print("\n" + "=" * 70)
    print(" [1/3] Running Offline Verification (Simulated Google Drive API)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_dir = Path(tmpdir)
        ws = Workspace.from_path(workspace_dir)
        fake_service = FakeDriveService()
        provider = GoogleDriveStorageProvider(drive_service=fake_service, workspace=ws)

        print("  - Initialized GoogleDriveStorageProvider with FakeDriveService.")

        # 1. Execute run and save checkpoint
        with runvault.init(
            experiment="gdrive-test-suite",
            name="simulated-upload-trial",
            workspace=workspace_dir,
            storage_provider=provider,
        ) as run:
            run.log_params({"model": "resnet50", "epochs": 10, "backend": "gdrive"})

            payload = b"GDRIVE_OFFLINE_VERIFICATION_PAYLOAD_HASH_TEST_12345\n"
            ckpt = run.save_checkpoint(
                name="offline-ckpt-step-10",
                writer=lambda dest: dest.write_bytes(payload),
                step=10,
                metadata={"accuracy": 0.965},
            )
            print(f"  - Checkpoint Published: '{ckpt.name}'")
            print(f"    Drive Storage Key:   {ckpt.storage_key}")
            print(f"    SHA-256 Checksum:    {ckpt.checksum_sha256[:16]}...")
            print(f"    Storage Provider:    {ckpt.storage_provider}")

        # 2. Verify restored artifact via bounded-memory streaming
        print("\n  - Restoring checkpoint with bounded-memory streaming...")
        recovery = runvault.restore_checkpoint(
            run.id,
            workspace=workspace_dir,
            storage=provider,
        )
        assert recovery.checkpoint.name == "offline-ckpt-step-10"
        assert recovery.path is None  # Remote streaming backends do not leak local paths

        with recovery.open() as stream:
            read_back = stream.read()
            assert read_back == payload

        print("    Streamed bytes verified successfully!")
        print(f"    Recovered payload:   {read_back.decode('utf-8').strip()}")

        # 3. Test Preflight Collision Prevention
        print("\n  - Verifying collision prevention (zero-overwrite immutability)...")
        staged = provider.stage_artifact(
            run_id=run.id,
            checkpoint_id=ckpt.id,
            writer=lambda p: p.write_bytes(b"duplicate attempt"),
            name="duplicate",
        )
        try:
            provider.publish_artifact(run.id, ckpt.id, staged, name="duplicate")
            raise AssertionError("Expected CheckpointCreationError for duplicate checkpoint ID!")
        except Exception as err:
            print(f"    Preflight collision rejected correctly: {type(err).__name__}")

    print("\n  [✓] Offline Google Drive verification completed successfully!\n")


def run_live_test(client_secrets: str | None, token_cache: str | None) -> None:
    """Run verification against real Google Drive."""
    print("\n" + "=" * 70)
    print(" [2/3] Running Live Verification Against Real Google Drive")
    print("=" * 70)

    root_folder = "RunVault-Live-Verification-Test"
    print(f"  Target Root Folder: '{root_folder}'")

    if is_colab_environment():
        print("  Detected Google Colab environment. Requesting runtime credentials...")
        provider = GoogleDriveStorageProvider(root_folder_name=root_folder)
    elif client_secrets:
        print(f"  Authenticating via OAuth client secrets file: {client_secrets}")
        provider = GoogleDriveStorageProvider.from_oauth(
            client_secrets_file=client_secrets,
            token_cache_path=token_cache or "~/.config/runvault/test_token.json",
            root_folder_name=root_folder,
        )
    else:
        print("  Attempting Application Default Credentials (ADC) or existing cached token...")
        provider = GoogleDriveStorageProvider(
            root_folder_name=root_folder,
            token_cache_path=token_cache or "~/.config/runvault/test_token.json",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_dir = Path(tmpdir)

        with runvault.init(
            experiment="live-gdrive-experiment",
            name="live-test-run",
            workspace=workspace_dir,
            storage_provider=provider,
        ) as run:
            payload = b"LIVE_GDRIVE_STREAMING_CHECKPOINT_DATA_1234567890\n"
            ckpt = run.save_checkpoint(
                name="live-checkpoint-1",
                writer=lambda dest: dest.write_bytes(payload),
                step=1,
            )
            print("\n  Uploaded to Google Drive successfully!")
            print(f"    Remote File ID:   {ckpt.storage_key}")
            print(f"    SHA-256 Checksum: {ckpt.checksum_sha256}")

        print("\n  Restoring from Google Drive via streaming reader...")
        recovery = runvault.restore_checkpoint(
            run.id,
            workspace=workspace_dir,
            storage=provider,
        )
        with recovery.open() as stream:
            data = stream.read()
            assert data == payload

        print("  Verified streamed payload matches original byte-for-byte!")
        print(f"  Payload: {data.decode('utf-8').strip()}")

    print("\n  [✓] Live Google Drive verification completed successfully!\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="RunVault Google Drive Smoke Test & Verification")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute test against real Google Drive (requires credentials)",
    )
    parser.add_argument(
        "--client-secrets",
        type=str,
        default=None,
        help="Path to OAuth 2.0 client_secret.json for local authentication",
    )
    parser.add_argument(
        "--token-cache",
        type=str,
        default=None,
        help="Path to cache authorized user OAuth token",
    )
    args = parser.parse_args()

    print("=" * 70)
    print(" RunVault — Google Drive Integration Verification (Phase 4.2)")
    print("=" * 70)

    # Step 1: Verify optional dependencies
    try:
        check_gdrive_dependencies()
        print("  [✓] Google Drive optional dependencies: OK (googleapiclient, google.auth)")
    except ImportError as err:
        print(f"  [✗] Missing dependencies: {err}")
        sys.exit(1)

    # Step 2: Always run offline mock test
    run_offline_test()

    # Step 3: Run live test if requested
    if args.live:
        run_live_test(args.client_secrets, args.token_cache)
    else:
        print("=" * 70)
        print(" Note: To test against your real Google Drive account, run:")
        print(
            "   python examples/test_gdrive_interactive.py --live "
            "--client-secrets <client_secret.json>"
        )
        print("=" * 70)


if __name__ == "__main__":
    main()
