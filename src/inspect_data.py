"""
Data Inspection & Tokenization Verification Script.
Purpose:
  1. Inspect raw dataset samples and verify prompt formatting.
  2. Verify tokenization behavior: raw token IDs, decoded strings, special tokens.
  3. Verify presence and positioning of EOS tokens (and BOS/PAD) to prevent silent training failures.
  4. Compute and display sequence length distributions (min, max, mean, median, 95th percentile).
"""

import os
import sys
import argparse
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect dataset and verify tokenization & EOS presence.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=os.environ.get("HF_MODEL_ID", "HuggingFaceTB/SmolLM2-135M"),
        help="Model ID or local path for tokenizer",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=os.environ.get("HF_DATASET", "timdettmers/openassistant-guanaco"),
        help="Hugging Face dataset identifier or local path",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to inspect (default: train)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field containing training text in dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample entries to inspect in detail",
    )
    parser.add_argument(
        "--max_stat_samples",
        type=int,
        default=1000,
        help="Number of samples to compute length statistics over",
    )
    return parser.parse_args()


def format_sample_with_eos(text: str, eos_token: str) -> str:
    """Ensure sample ends cleanly with the EOS token if not already present."""
    text = text.strip()
    if not text.endswith(eos_token):
        text = text + eos_token
    return text


def inspect_dataset(args):
    print("=" * 80)
    print("ML INNER-LOOP STEP 1: RAW TOKEN & EOS VERIFICATION")
    print("=" * 80)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading Tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Configure PAD token if missing
    if tokenizer.pad_token is None:
        print("[INFO] pad_token is None. Setting tokenizer.pad_token = tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Tokenizer Vocab Size: {tokenizer.vocab_size}")
    print(f"EOS Token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"BOS Token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"PAD Token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"Chat Template: {tokenizer.chat_template is not None}")
    print("-" * 80)

    print(f"Loading Dataset: {args.dataset_name} (split: {args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)
    total_samples = len(dataset)
    print(f"Total entries in split: {total_samples}")
    print("-" * 80)

    # Verify presence of text field
    sample_first = dataset[0]
    if args.text_field not in sample_first:
        available_fields = list(sample_first.keys())
        raise ValueError(
            f"Field '{args.text_field}' not found in dataset! Available fields: {available_fields}"
        )

    # Inspect sample entries
    print(f"\n--- Detailed Inspection of First {args.num_samples} Samples ---")
    eos_failures = 0

    for idx in range(min(args.num_samples, total_samples)):
        raw_text = dataset[idx][args.text_field]
        # In causal LM fine-tuning, training strings must terminate with EOS
        formatted_text = format_sample_with_eos(raw_text, tokenizer.eos_token)

        # Tokenize without truncation to inspect exact sequence
        tokenized = tokenizer(
            formatted_text,
            add_special_tokens=True,
            return_attention_mask=True,
        )
        input_ids = tokenized["input_ids"]
        decoded_tail = tokenizer.decode(input_ids[-5:])

        print(f"\n[Sample #{idx + 1}]")
        print(f"Raw Text Preview (first 120 chars): {repr(raw_text[:120])}...")
        print(f"Raw Text Tail (last 60 chars): {repr(raw_text[-60:])}")
        print(f"Token Length: {len(input_ids)}")
        print(f"Tail 5 Token IDs: {input_ids[-5:]}")
        print(f"Tail 5 Decoded: {repr(decoded_tail)}")

        # Verification check: Last token must be EOS
        has_eos_at_end = (input_ids[-1] == tokenizer.eos_token_id)
        if has_eos_at_end:
            print("  --> [OK] Sequence terminates with explicit EOS token.")
        else:
            print(f"  --> [FAIL] Sequence does NOT terminate with EOS token! End token: {input_ids[-1]}")
            eos_failures += 1

    # Statistical verification across sample window
    print("\n" + "=" * 80)
    print(f"Computing Token Length Statistics (Sample size: {min(args.max_stat_samples, total_samples)})")
    print("=" * 80)

    stat_subset = dataset.select(range(min(args.max_stat_samples, total_samples)))
    lengths = []
    missing_eos_count = 0

    for item in stat_subset:
        formatted_text = format_sample_with_eos(item[args.text_field], tokenizer.eos_token)
        ids = tokenizer.encode(formatted_text, add_special_tokens=True)
        lengths.append(len(ids))
        if ids[-1] != tokenizer.eos_token_id:
            missing_eos_count += 1

    lengths = np.array(lengths)
    p50 = float(np.percentile(lengths, 50))
    p95 = float(np.percentile(lengths, 95))
    p99 = float(np.percentile(lengths, 99))

    print(f"Minimum Token Length: {np.min(lengths)}")
    print(f"Maximum Token Length: {np.max(lengths)}")
    print(f"Mean Token Length:    {np.mean(lengths):.2f}")
    print(f"Median (p50) Length:  {p50:.1f}")
    print(f"95th Percentile:      {p95:.1f}")
    print(f"99th Percentile:      {p99:.1f}")
    print(f"Sequences Missing EOS: {missing_eos_count}/{len(lengths)}")

    # Verification assertions
    assert missing_eos_count == 0, (
        f"CRITICAL: {missing_eos_count} sequences failed the EOS termination check! "
        "A missing EOS will prevent generation from terminating cleanly."
    )
    assert len(lengths) > 0, "No sequences were processed."
    assert np.min(lengths) > 0, "Encountered empty sequence (0 tokens)."

    print("\n" + "#" * 80)
    print("# [PASS] TOKENIZATION & EOS VERIFICATION COMPLETED SUCCESSFULLY")
    print(f"# Recommended max_seq_length for training: {int(np.ceil(p95 / 64) * 64)}")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    args = parse_args()
    inspect_dataset(args)
