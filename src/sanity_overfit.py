"""
Single-Batch Overfitting Sanity Check.
Purpose:
  Prove gradients flow and LoRA adapters train properly before spinning up cloud compute.
  Overfits a single batch over 40-50 iterations using AdamW, verifying a steep drop in loss.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Run single-batch overfitting sanity check.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=os.environ.get("HF_MODEL_ID", "HuggingFaceTB/SmolLM2-135M"),
        help="Base model ID or path",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=os.environ.get("HF_DATASET", "timdettmers/openassistant-guanaco"),
        help="Dataset name or path (optional)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of optimization iterations (default: 50)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate for AdamW (default: 5e-4)",
    )
    parser.add_argument(
        "--loss_threshold",
        type=float,
        default=0.25,
        help="Target loss threshold to consider overfit successful (default: 0.25)",
    )
    parser.add_argument(
        "--reduction_target",
        type=float,
        default=0.80,
        help="Target relative loss reduction (e.g. 0.80 = 80%% drop)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    return parser.parse_args()


def get_single_batch(tokenizer, dataset_name=None, batch_size=2, max_length=128, device="cpu"):
    """
    Fetch a single batch of 2 samples.
    Attempts to read from the specified dataset, or falls back to synthetic instruction pairs.
    """
    samples = []
    if dataset_name:
        try:
            from datasets import load_dataset
            print(f"Attempting to load 2 samples from dataset '{dataset_name}'...")
            ds = load_dataset(dataset_name, split="train")
            for i in range(min(batch_size, len(ds))):
                text = ds[i]["text"]
                if not text.endswith(tokenizer.eos_token):
                    text += tokenizer.eos_token
                samples.append(text)
        except Exception as e:
            print(f"[NOTE] Could not load dataset '{dataset_name}' directly ({e}). Using synthetic instruction batch.")

    if not samples:
        # High-signal synthetic instruction samples for sanity overfit
        samples = [
            (
                "### Human: What is the capital of France?\n"
                "### Assistant: The capital of France is Paris." + tokenizer.eos_token
            ),
            (
                "### Human: Explain QLoRA in one sentence.\n"
                "### Assistant: QLoRA quantizes the base model to 4-bit while training high-precision LoRA adapters."
                + tokenizer.eos_token
            ),
        ]

    print(f"Batch contains {len(samples)} sample prompts.")
    for idx, s in enumerate(samples):
        print(f"  [Sample {idx+1}]: {repr(s)}")

    # Tokenize batch with padding and truncation
    encoded = tokenizer(
        samples,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    # For causal LM, labels are identical to input_ids; mask out pad tokens with -100
    labels = input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_sanity_overfit(args):
    print("=" * 80)
    print("ML INNER-LOOP STEP 2: SINGLE-BATCH OVERFITTING SANITY CHECK")
    print("=" * 80)

    # Determine execution device and dtype
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"[Device] Using CUDA GPU: {torch.cuda.get_device_name(0)} (dtype: {torch_dtype})")
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32
        print(f"[Device] Using CPU fallback (dtype: {torch_dtype})")

    # Import transformers and PEFT
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"Loading Tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading Model: {args.model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    base_model.to(device)

    # Configure LoRA
    # SmolLM2 uses standard LLaMA-style projections: q_proj, k_proj, v_proj, o_proj
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )

    print("Injecting LoRA adapters...")
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.train()

    # Extract single batch
    batch = get_single_batch(
        tokenizer,
        dataset_name=args.dataset_name,
        batch_size=2,
        max_length=128,
        device=device,
    )

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print("-" * 80)
    print(f"Starting single-batch optimization for {args.iterations} iterations (lr={args.learning_rate})...")
    print("-" * 80)

    loss_history = []
    initial_loss = None

    for step in range(1, args.iterations + 1):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # Verify gradient flow
        loss.backward()

        # Check for non-zero gradients on trainable parameters
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grad_norms or all(gn == 0.0 for gn in grad_norms):
            raise RuntimeError(f"Step {step}: Gradients failed to flow into trainable LoRA parameters!")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = loss.item()
        loss_history.append(current_loss)

        if initial_loss is None:
            initial_loss = current_loss

        if step % 5 == 0 or step == 1 or step == args.iterations:
            reduction = (initial_loss - current_loss) / initial_loss * 100.0
            print(f"Step {step:02d}/{args.iterations} | Loss: {current_loss:.4f} | Reduction: {reduction:.1f}%")

    final_loss = loss_history[-1]
    total_reduction = (initial_loss - final_loss) / initial_loss

    print("=" * 80)
    print(f"Optimization Summary:")
    print(f"  Initial Loss:    {initial_loss:.4f}")
    print(f"  Final Loss:      {final_loss:.4f}")
    print(f"  Total Reduction: {total_reduction * 100:.2f}%")
    print("=" * 80)

    # Verify overfit criteria
    is_success = (final_loss < args.loss_threshold) or (total_reduction >= args.reduction_target)
    
    if not is_success:
        print(f"FAIL: Loss did not converge sufficiently!")
        print(f"Final loss {final_loss:.4f} >= threshold {args.loss_threshold} AND reduction {total_reduction*100:.1f}% < {args.reduction_target*100:.1f}%")
        sys.exit(1)

    print("\n" + "#" * 80)
    print("# [PASS] SANITY OVERFIT TEST PASSED: GRADIENT FLOW & BACKPROP PROVEN!")
    print("# LoRA adapters converged successfully on 1 batch.")
    print("#" * 80 + "\n")
    return 0


if __name__ == "__main__":
    args = parse_args()
    run_sanity_overfit(args)
