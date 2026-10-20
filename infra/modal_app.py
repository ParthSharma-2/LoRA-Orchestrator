"""
infra/modal_app.py
==================
Step 4: Serverless GPU Container & Execution via Modal

Defines the serverless cloud infrastructure for fine-tuning SmolLM2-135M with QLoRA.
- Modal App definition (`smollm2-qlora-pipeline`).
- Container image setup with PyTorch, CUDA, bitsandbytes, and Hugging Face libraries.
- NVIDIA A10G (or T4) GPU allocation.
- Secret mounting for Hugging Face Hub (HF_TOKEN) and Weights & Biases (WANDB_API_KEY).
- Persistent volume mounting for checkpoints and merged weights.
- Streaming logs and remote execution orchestration.
- Local CLI entrypoint for `modal run infra/modal_app.py`.
"""

import os
import sys
import subprocess
from pathlib import Path

# Load local .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import modal

# ---------------------------------------------------------------------------
# App & Resource Configuration
# ---------------------------------------------------------------------------
APP_NAME = "smollm2-qlora-pipeline"
DEFAULT_GPU = os.environ.get("MODAL_GPU", "A10G")
VOLUME_NAME = "smollm2-qlora-checkpoints"

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SRC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Modal Image Definition
# ---------------------------------------------------------------------------
# Debian slim with Python 3.11 and the complete QLoRA training stack
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "transformers>=4.44.0",
        "peft>=0.12.0",
        "trl>=0.9.6",
        "bitsandbytes>=0.43.0",
        "datasets>=2.20.0",
        "accelerate>=0.33.0",
        "wandb>=0.16.0",
        "scipy>=1.11.0",
        "sentencepiece>=0.2.0",
        "huggingface_hub>=0.22.0",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": "/root",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(str(SRC_DIR), remote_path="/root/src")
)

# ---------------------------------------------------------------------------
# Modal App, Secrets, and Volumes
# ---------------------------------------------------------------------------
app = modal.App(name=APP_NAME)

# Construct secrets: prioritize local .env / environment variables
secrets = []
env_secrets = {}
for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "WANDB_PROJECT"]:
    val = os.environ.get(key)
    if val:
        env_secrets[key] = val

if env_secrets:
    secrets.append(modal.Secret.from_dict(env_secrets))
elif os.environ.get("USE_MODAL_NAMED_SECRETS", "").lower() in ("1", "true", "yes"):
    # Optional fallback to pre-configured Modal dashboard named secrets
    secrets.extend([
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret"),
    ])

# Persistent volume for checkpoints and artifacts
checkpoint_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ---------------------------------------------------------------------------
# Remote Training Function
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    secrets=secrets,
    volumes={"/root/checkpoints": checkpoint_volume},
    timeout=7200,  # 2 hours max execution
)
def train_remote(
    model_id: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "timdettmers/openassistant-guanaco",
    output_dir: str = "/root/checkpoints/smollm2-qlora",
    num_train_epochs: float = 3.0,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    push_to_hub: bool = False,
    merge_and_push: bool = False,
    hub_model_id: str = "",
    report_to: str = "wandb",
):
    """Executes the QLoRA training script on a serverless GPU in Modal.

    Streams live stdout/stderr logs and persists checkpoints to the Modal Volume.
    """
    print("=" * 70)
    print("🚀 INITIALIZING MODAL SERVERLESS QLORA FINE-TUNING")
    print(f"📦 Model ID        : {model_id}")
    print(f"📊 Dataset         : {dataset_name}")
    print(f"📁 Output Dir      : {output_dir}")
    print(f"⏱️  Epochs          : {num_train_epochs}")
    print(f"📦 Batch Size      : {per_device_train_batch_size} (Grad Accum: {gradient_accumulation_steps})")
    print(f"📈 Learning Rate   : {learning_rate}")
    print(f"📏 Max Seq Length  : {max_seq_length}")
    print(f"🌐 Push to Hub     : {push_to_hub} (Target: {hub_model_id or 'None'})")
    print(f"📡 Telemetry       : {report_to}")
    print("=" * 70)

    # Verify CUDA availability on the remote GPU
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
        vram = (
            f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB"
            if cuda_available
            else "N/A"
        )
        print(f"🖥️  GPU Hardware    : {gpu_name} (Total VRAM: {vram})")
        if not cuda_available:
            raise RuntimeError("CUDA is not available on remote runner! Check Modal GPU specification.")
    except ImportError:
        print("⚠️  Warning: PyTorch not found in pre-check.")

    # Check for authentication tokens
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")

    if hf_token:
        print("🔑 Hugging Face token detected.")
    else:
        print("ℹ️  No HF token detected. Private repos & push_to_hub may fail unless public.")

    if wandb_key:
        print("🔑 Weights & Biases API key detected. Run will be logged to W&B.")
    else:
        print("ℹ️  No W&B API key detected. If report_to='wandb', training will proceed in offline/fallback mode.")
        if report_to == "wandb":
            os.environ["WANDB_MODE"] = "offline"

    # Assemble training CLI arguments
    script_path = Path("/root/src/train.py")
    if not script_path.exists():
        raise FileNotFoundError(
            f"Training script not found at {script_path}. Ensure src/train.py exists locally before launching."
        )

    cmd = [
        sys.executable,
        "-u",
        str(script_path),
        "--model_id", str(model_id),
        "--dataset_name", str(dataset_name),
        "--output_dir", str(output_dir),
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(learning_rate),
        "--max_seq_length", str(max_seq_length),
        "--report_to", str(report_to),
    ]

    if merge_and_push:
        cmd.append("--merge_and_push")
        if hub_model_id:
            cmd.extend(["--hub_model_id", str(hub_model_id)])
    elif push_to_hub:
        cmd.append("--push_to_hub")
        if hub_model_id:
            cmd.extend(["--hub_model_id", str(hub_model_id)])

    print(f"\n▶️  Executing command: {' '.join(cmd)}\n")

    # Launch subprocess with streaming unbuffered stdout/stderr
    process = subprocess.Popen(
        cmd,
        cwd="/root",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"❌ Training failed with process exit code {return_code}")

    print("\n💾 Committing persistent volume changes...")
    checkpoint_volume.commit()
    print("✅ Volume committed successfully! Checkpoints are saved to persistent storage.")
    return {"status": "success", "output_dir": output_dir, "model_id": model_id}


# ---------------------------------------------------------------------------
# Local CLI Entrypoint
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    model_id: str = "HuggingFaceTB/SmolLM2-135M",
    dataset_name: str = "timdettmers/openassistant-guanaco",
    output_dir: str = "/root/checkpoints/smollm2-qlora",
    epochs: float = 3.0,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    push_to_hub: bool = False,
    merge_and_push: bool = False,
    hub_model_id: str = "",
    report_to: str = "wandb",
):
    """Local entrypoint triggered by `modal run infra/modal_app.py`.

    Accepts CLI flags:
        --epochs (float)
        --batch-size (int)
        --gradient-accumulation-steps (int)
        --learning-rate (float)
        --max-seq-length (int)
        --model-id (str)
        --dataset-name (str)
        --output-dir (str)
        --push-to-hub (flag)
        --hub-model-id (str)
        --report-to (str)
    """
    print(f"🎯 Dispatching QLoRA training job to Modal ({DEFAULT_GPU})...")
    result = train_remote.remote(
        model_id=model_id,
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        push_to_hub=push_to_hub,
        merge_and_push=merge_and_push,
        hub_model_id=hub_model_id,
        report_to=report_to,
    )
    print(f"🎉 Job completed with result: {result}")
