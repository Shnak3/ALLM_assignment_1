from pathlib import Path
import subprocess
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import sys

checkpoint_dir = (
    Path.home()
    / ".cache"
    / "nanochat"
    / "base_checkpoints"
    / "d4"
)

output_csv = "bpb_results.csv"
output_plot = "bpb_curves.png"

metadata_files = sorted(checkpoint_dir.glob("meta_*.json"))

results = []

for meta_path in metadata_files:

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    step = meta["step"]
    val_bpb = meta["val_bpb"]

    print(f"\nEvaluating checkpoint step {step}...")

    process = subprocess.run(
        [sys.executable, "-m", "scripts.base_eval", "--eval=bpb", f"--model-tag=d4", f"--step={step}"], capture_output=True, text=True,)

    output = process.stdout + "\n" + process.stderr

    train_match = re.search(r"train bpb:\s*([0-9.]+)", output, re.IGNORECASE,)

    train_bpb = float(train_match.group(1))
    results.append({"step": step, "train_bpb": train_bpb, "val_bpb": val_bpb,})

    print(
        f"Step {step}: "
        f"train BPB = {train_bpb:.6f}, "
        f"val BPB = {val_bpb:.6f}"
    )

df = pd.DataFrame(results)
df = df.sort_values("step")

df.to_csv(output_csv, index=False)

plt.figure(figsize=(8, 5))

plt.plot(df["step"], df["train_bpb"], marker="o", label="Training BPB",)

plt.plot(df["step"], df["val_bpb"], marker="o", label="Validation BPB",)

plt.xlabel("Training step")
plt.ylabel("Bits per byte (BPB)")
plt.title("Training and validation BPB")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(output_plot, dpi=300)
plt.show()