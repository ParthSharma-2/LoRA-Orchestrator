"""
Local Inference Script for Fine-Tuned SmolLM2-135M.
Runs efficiently on Intel i7 CPU (or CUDA GPU if available).
"""

import os
import sys
import argparse
import torch

try:
    import torchao
    from packaging import version
    if version.parse(getattr(torchao, "__version__", "0.0.0")) < version.parse("0.16.0"):
        sys.modules["torchao"] = None
except Exception:
    pass

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
    has_peft = True
except ImportError:
    has_peft = False


def parse_args():
    parser = argparse.ArgumentParser(description="Test fine-tuned model inference locally.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolLM2-135M",
        help="Base model ID or merged model ID on Hugging Face Hub (or local directory)",
    )
    parser.add_argument(
        "--adapter_id",
        type=str,
        default=None,
        help="Optional LoRA adapter ID (Hub repo ID or local './output' directory)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="### Human: What are the benefits of serverless computing for machine learning?\n### Assistant:",
        help="Input prompt for the model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling probability",
    )
    return parser.parse_args()


def generate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("🤖 LOCAL INFERENCE RUNNER")
    print(f"Device      : {device.upper()}")
    print(f"Model ID    : {args.model_id}")
    if args.adapter_id:
        print(f"Adapter ID  : {args.adapter_id}")
    print("=" * 70)

    # 1. Load Tokenizer
    tokenizer_source = args.adapter_id if args.adapter_id else args.model_id
    print(f"Loading tokenizer from: {tokenizer_source}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Model
    print(f"Loading model weights from: {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True,
    )

    # 3. Attach LoRA adapter if provided
    if args.adapter_id:
        if not has_peft:
            raise ImportError("peft library is required to load LoRA adapters. Run: pip install peft")
        print(f"Attaching LoRA adapter from: {args.adapter_id}...")
        model = PeftModel.from_pretrained(model, args.adapter_id)

    model.eval()

    # 4. Run Generation
    print("\n[Input Prompt]:")
    print(args.prompt)
    print("\n[Generating response...]")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode and print generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n" + "=" * 70)
    print("✨ GENERATED OUTPUT:")
    print("=" * 70)
    print(generated_text)
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    generate(args)
