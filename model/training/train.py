# Swaraaha - Training Orchestrator
# Master script that auto-detects system resources and runs training pipelines.
#
# Usage:
#   python -m model.training.train                  # Run all pipelines
#   python -m model.training.train --pipelines cls  # Classification only
#   python -m model.training.train --pipelines loc  # CNN localization only
#   python -m model.training.train --pipelines wav2vec  # Wav2Vec2 localization only
#   python -m model.training.train --pipelines cls loc  # Multiple

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemInfo:
    """Auto-detected system resources."""
    cpu_count: int = field(default_factory=os.cpu_count)
    has_gpu: bool = False
    gpu_name: str = "None"
    gpu_memory_gb: float = 0.0

    def detect(self):
        """Detect CPU and GPU resources."""
        self.cpu_count = os.cpu_count() or 1

        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            if self.has_gpu:
                self.gpu_name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_mem
                self.gpu_memory_gb = mem / (1024 ** 3)
        except ImportError:
            pass

        return self

    def optimal_batch_size(self, pipeline: str) -> int:
        """Suggest optimal batch size based on available resources."""
        if pipeline == "cls":
            if self.has_gpu and self.gpu_memory_gb >= 8:
                return 16
            elif self.has_gpu:
                return 8
            return 4
        elif pipeline == "loc":
            if self.has_gpu and self.gpu_memory_gb >= 8:
                return 8
            elif self.has_gpu:
                return 4
            return 2
        elif pipeline == "wav2vec":
            if self.has_gpu and self.gpu_memory_gb >= 8:
                return 4
            elif self.has_gpu:
                return 2
            return 1
        return 4

    def optimal_num_workers(self) -> int:
        """Suggest optimal DataLoader workers."""
        return self.cpu_count

    def __str__(self) -> str:
        lines = [
            f"  CPU: {self.cpu_count} cores",
            f"  GPU: {'Yes' if self.has_gpu else 'No'}",
        ]
        if self.has_gpu:
            lines.append(f"    Device: {self.gpu_name}")
            lines.append(f"    Memory: {self.gpu_memory_gb:.1f} GB")
        return "\n".join(lines)


CLASS_NAMES = ["prolongation", "block", "soundrep", "wordrep", "interjection"]

PIPELINE_SCRIPTS = {
    "cls": "model.training.train_classifier",
    "loc": "model.training.train_localizer",
    "wav2vec": "model.training.train_wav2vec2_localizer",
}


def run_pipeline(
    pipeline: str,
    data_dir: str,
    output_dir: str,
    system: SystemInfo,
    extra_args: list[str] | None = None,
) -> int:
    """Run a single training pipeline.

    Args:
        pipeline: One of 'cls', 'loc', 'wav2vec'.
        data_dir: Path to training data directory.
        output_dir: Path to save model weights.
        system: Detected system info for parameter tuning.
        extra_args: Additional CLI arguments to pass.

    Returns:
        Exit code (0 = success).
    """
    script = PIPELINE_SCRIPTS[pipeline]
    batch_size = system.optimal_batch_size(pipeline)
    num_workers = system.optimal_num_workers()

    if pipeline == "cls":
        failed = []
        for cls_name in CLASS_NAMES:
            print(f"\n{'='*60}")
            print(f"Training classifier: {cls_name}")
            print(f"{'='*60}")

            cmd = [
                sys.executable, "-m", script,
                "--class_name", cls_name,
                "--data_dir", data_dir,
                "--output_dir", output_dir,
                "--batch_size", str(batch_size),
                "--num_workers", str(num_workers),
            ]
            if extra_args:
                cmd.extend(extra_args)

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  WARNING: {cls_name} classifier failed (exit code {result.returncode})")
                failed.append(cls_name)

        if failed:
            print(f"\nFailed classifiers: {', '.join(failed)}")
            return 1
        return 0

    else:
        cmd = [
            sys.executable, "-m", script,
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--batch_size", str(batch_size),
            "--num_workers", str(num_workers),
        ]
        if extra_args:
            cmd.extend(extra_args)

        result = subprocess.run(cmd)
        return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Swaraaha Training Orchestrator — auto-detects system and runs pipelines."
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=list(PIPELINE_SCRIPTS.keys()),
        default=list(PIPELINE_SCRIPTS.keys()),
        help="Which pipelines to run (default: all).",
    )
    parser.add_argument("--data_dir", type=str, default="data/train", help="Training data directory.")
    parser.add_argument("--output_dir", type=str, default="model/weights", help="Output directory for weights.")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be run without executing.")
    parser.add_argument("extra_args", nargs="*", help="Additional args passed to each pipeline script.")
    args = parser.parse_args()

    system = SystemInfo().detect()

    print("Swaraaha Training Orchestrator")
    print(f"{'='*40}")
    print(f"System:")
    print(system)
    print(f"\nPipelines: {', '.join(args.pipelines)}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")

    for pipeline in args.pipelines:
        bs = system.optimal_batch_size(pipeline)
        nw = system.optimal_num_workers()
        print(f"  {pipeline}: batch_size={bs}, num_workers={nw}")

    if args.dry_run:
        print("\nDry run — no training executed.")
        return

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"\nError: Data directory not found at {data_path}")
        print("Run 'python -m model.data.setup' first to prepare training data.")
        sys.exit(1)

    start = time.time()
    results = {}

    for pipeline in args.pipelines:
        print(f"\n{'#'*60}")
        print(f"# Running: {pipeline}")
        print(f"{'#'*60}")

        code = run_pipeline(
            pipeline=pipeline,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            system=system,
            extra_args=args.extra_args,
        )
        results[pipeline] = code

    elapsed = time.time() - start
    print(f"\n{'='*40}")
    print(f"Training complete in {elapsed:.1f}s")
    for pipeline, code in results.items():
        status = "OK" if code == 0 else "FAILED"
        print(f"  {pipeline}: {status}")

    if any(code != 0 for code in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
