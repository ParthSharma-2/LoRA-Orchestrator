"""
Production QLoRA / LoRA SFT Training Script.
Purpose:
  1. Fine-tune SmolLM2-135M (or target model) using SFTTrainer and PEFT.
  2. Support 4-bit NF4 QLoRA via bitsandbytes with automatic fallback to standard LoRA.
  3. Support Weights & Biases telemetry.
  4. Save adapter checkpoints and optionally merge with base model and push to Hugging Face Hub.
"""

import os
import sys
import argparse
import logging
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("sft_trainer")


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA / LoRA Fine-Tuning using TRL SFTTrainer")
    
    # Model & Dataset arguments
    parser.add_argument(
        "--model_id",
        type=str,
        default=os.environ.get("HF_MODEL_ID", "HuggingFaceTB/SmolLM2-135M"),
        help="Base model ID on Hugging Face Hub or local path",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=os.environ.get("HF_DATASET", "timdettmers/openassistant-guanaco"),
        help="Dataset name or path",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split for training",
    )
    parser.add_argument(
        "--dataset_text_field",
        type=str,
        default="text",
        help="Field name containing training text",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "./output"),
        help="Directory to save adapter checkpoints",
    )
    
    # Training hyperparameters
    parser.add_argument("--epochs", "--num_train_epochs", dest="epochs", type=float, default=3.0, help="Total training epochs")
    parser.add_argument("--batch_size", "--per_device_train_batch_size", dest="batch_size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every X steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every X steps")
    
    # LoRA / QLoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target linear modules for LoRA",
    )
    parser.add_argument(
        "--disable_qlora",
        action="store_true",
        help="Disable 4-bit quantization and use standard LoRA",
    )
    
    # Telemetry
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        help="Reporting backend: 'wandb', 'tensorboard', or 'none'",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "smollm2-qlora"),
        help="Weights & Biases project name",
    )
    
    # Hugging Face Hub integration
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push adapter to Hugging Face Hub",
    )
    parser.add_argument(
        "--merge_and_push",
        action="store_true",
        help="Merge LoRA adapter into base model and push full weights to Hub",
    )
    parser.add_argument(
        "--hub_repo_id",
        "--hub_model_id",
        dest="hub_repo_id",
        type=str,
        default=os.environ.get("HF_REPO_ID", None),
        help="Target repo ID on Hugging Face Hub (e.g. username/smollm2-qlora)",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face API token",
    )
    
    return parser.parse_args()


def setup_telemetry(args):
    """Set up reporting environments like Weights & Biases."""
    if args.report_to == "wandb":
        wandb_key = os.environ.get("WANDB_API_KEY")
        if not wandb_key:
            logger.warning("report_to is 'wandb' but WANDB_API_KEY is not set. Falling back to 'none'.")
            args.report_to = "none"
        else:
            os.environ["WANDB_PROJECT"] = args.wandb_project
            logger.info(f"Weights & Biases telemetry enabled. Project: {args.wandb_project}")


def get_quantization_and_model_config(args):
    """Configure model loading with 4-bit NF4 QLoRA or standard precision LoRA fallback."""
    cuda_available = torch.cuda.is_available()
    can_use_qlora = cuda_available and (not args.disable_qlora)

    bnb_config = None
    if can_use_qlora:
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            logger.info(f"Configured 4-bit NF4 QLoRA (compute_dtype: {compute_dtype}).")
        except Exception as e:
            logger.warning(f"BitsAndBytesConfig failed ({e}). Falling back to non-quantized LoRA.")
            bnb_config = None
    else:
        if not cuda_available:
            logger.info("CUDA not detected; running in CPU standard precision mode.")
        else:
            logger.info("QLoRA explicitly disabled; running in standard 16-bit LoRA mode.")

    return bnb_config


def formatting_prompts_func(example, tokenizer, text_field="text"):
    """
    Format examples to ensure the sequence ends with EOS token.
    Prevents generation infinite loops caused by missing EOS.
    """
    text = example[text_field].strip()
    if not text.endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token
    return text


def train():
    args = parse_args()
    logger.info("Starting SFT fine-tuning pipeline...")
    logger.info(f"Arguments: {vars(args)}")

    setup_telemetry(args)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
    from trl import SFTTrainer

    # Check for SFTConfig vs TrainingArguments compatibility across TRL versions
    try:
        from trl import SFTConfig
        has_sft_config = True
    except ImportError:
        from transformers import TrainingArguments
        has_sft_config = False

    # 1. Initialize Tokenizer
    logger.info(f"Loading Tokenizer from: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        logger.info("Setting tokenizer.pad_token = tokenizer.eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # 2. Load Dataset
    logger.info(f"Loading Dataset: {args.dataset_name} (split: {args.dataset_split})")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    logger.info(f"Dataset loaded with {len(dataset)} examples.")

    # Apply EOS formatting check
    dataset = dataset.map(lambda ex: {args.dataset_text_field: formatting_prompts_func(ex, tokenizer, args.dataset_text_field)})

    # 3. Load Model with QLoRA configuration
    bnb_config = get_quantization_and_model_config(args)
    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if torch.cuda.is_available() else torch.float32)
    )

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    if device_map:
        model_kwargs["device_map"] = device_map
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config

    logger.info(f"Loading Model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Disable KV cache during training to allow gradient checkpointing
    model.config.use_cache = False

    # 4. LoRA Configuration
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )

    # 5. Training Arguments / SFTConfig
    cuda_available = torch.cuda.is_available()
    use_gradient_checkpointing = cuda_available  # Disable on CPU for speed and autograd stability

    if use_gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    common_training_args = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=(cuda_available and not torch.cuda.is_bf16_supported()),
        bf16=(cuda_available and torch.cuda.is_bf16_supported()),
        use_cpu=(not cuda_available),
        gradient_checkpointing=use_gradient_checkpointing,
        report_to=args.report_to,
    )

    import inspect
    trainer_init_params = inspect.signature(SFTTrainer.__init__).parameters

    if has_sft_config:
        sft_config_params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
        
        # Merge all training configurations
        candidate_args = dict(**common_training_args)
        
        # Handle max length parameter naming across TRL versions
        if "max_length" in sft_config_params:
            candidate_args["max_length"] = args.max_seq_length
        elif "max_seq_length" in sft_config_params:
            candidate_args["max_seq_length"] = args.max_seq_length
        
        if "dataset_text_field" in sft_config_params:
            candidate_args["dataset_text_field"] = args.dataset_text_field
        if "packing" in sft_config_params:
            candidate_args["packing"] = False

        # Filter strictly to arguments accepted by SFTConfig in this TRL version
        filtered_sft_args = {k: v for k, v in candidate_args.items() if k in sft_config_params}
        
        # If warmup_steps is supported but warmup_ratio is not, convert to steps
        if "warmup_steps" in sft_config_params and "warmup_ratio" not in sft_config_params:
            total_steps = max(1, (len(dataset) // (args.batch_size * args.gradient_accumulation_steps)) * int(args.epochs))
            filtered_sft_args["warmup_steps"] = max(1, int(total_steps * args.warmup_ratio))

        training_args = SFTConfig(**filtered_sft_args)
        trainer_kwargs = {
            "model": model,
            "train_dataset": dataset,
            "peft_config": peft_config,
            "args": training_args,
        }
    else:
        from transformers import TrainingArguments
        training_args = TrainingArguments(**common_training_args)
        trainer_kwargs = {
            "model": model,
            "train_dataset": dataset,
            "peft_config": peft_config,
            "args": training_args,
            "max_seq_length": args.max_seq_length,
            "dataset_text_field": args.dataset_text_field,
            "packing": False,
        }

    # Pass tokenizer via processing_class or tokenizer depending on TRL version
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # 6. Initialize Trainer & Train
    logger.info("Initializing SFTTrainer...")
    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting training loop...")
    trainer.train()

    # 7. Save Adapter & Tokenizer
    logger.info(f"Saving fine-tuned LoRA adapter to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 8. Push to Hugging Face Hub (Adapter or Merged)
    hub_repo_id = args.hub_repo_id or os.environ.get("HF_REPO_ID")
    if hub_repo_id:
        hub_repo_id = hub_repo_id.replace("<", "").replace(">", "").strip()
    hub_token = args.hub_token or os.environ.get("HF_TOKEN")

    if args.merge_and_push:
        if not hub_repo_id:
            raise ValueError("Merging and pushing to Hub requested, but --hub_repo_id or HF_REPO_ID is not provided.")
        logger.info(f"Merging LoRA adapter with base model and pushing to {hub_repo_id}...")
        merge_and_push_to_hub(
            base_model_id=args.model_id,
            adapter_dir=args.output_dir,
            hub_repo_id=hub_repo_id,
            hub_token=hub_token,
        )
    elif args.push_to_hub:
        if not hub_repo_id:
            raise ValueError("Pushing adapter to Hub requested, but --hub_repo_id or HF_REPO_ID is not provided.")
        logger.info(f"Pushing LoRA adapter checkpoint to {hub_repo_id}...")
        trainer.model.push_to_hub(hub_repo_id, token=hub_token)
        tokenizer.push_to_hub(hub_repo_id, token=hub_token)

    logger.info("Training pipeline completed successfully!")


def merge_and_push_to_hub(base_model_id: str, adapter_dir: str, hub_repo_id: str, hub_token: str = None):
    """Load base model in full precision, merge LoRA weights, and upload to Hub."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Loading base model '{base_model_id}' for merge...")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    logger.info(f"Loading PEFT adapter from '{adapter_dir}'...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    logger.info("Merging adapter into base model weights...")
    merged_model = peft_model.merge_and_unload()

    logger.info(f"Pushing merged model to Hugging Face Hub: {hub_repo_id}...")
    merged_model.push_to_hub(hub_repo_id, token=hub_token, safe_serialization=True)
    tokenizer.push_to_hub(hub_repo_id, token=hub_token)
    logger.info("Merged model successfully uploaded to Hub.")


if __name__ == "__main__":
    train()
