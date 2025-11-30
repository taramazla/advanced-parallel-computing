"""
Fine-tuning LLM with LoRA (Low-Rank Adaptation)
Supports various base models: GPT-2, LLaMA, Mistral, etc.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA")
    parser.add_argument("--model_name", type=str, default="gpt2",
                        help="Base model name (e.g., gpt2, meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--dataset_name", type=str, default="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                        help="Dataset name from HuggingFace")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration (optional)")
    parser.add_argument("--output_dir", type=str, default="./lora_outputs",
                        help="Output directory for model checkpoints")
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA attention dimension (rank)")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout probability")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--use_4bit", action="store_true",
                        help="Use 4-bit quantization (QLoRA)")
    parser.add_argument("--use_8bit", action="store_true",
                        help="Use 8-bit quantization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log metrics every X steps")
    return parser.parse_args()


def prepare_dataset(dataset_name, dataset_config, tokenizer, max_length):
    """Load and prepare dataset for training"""
    if dataset_config:
        print(f"Loading dataset: {dataset_name}/{dataset_config}")
        try:
            dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
        except:
            print(f"Failed to load with config, trying without config...")
            dataset = load_dataset(dataset_name, trust_remote_code=True)
    else:
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    
    def format_instruction(examples):
        """Format customer service data as instruction-following"""
        texts = []
        # Handle different column names in the dataset
        if "question" in examples and "answer" in examples:
            for q, a in zip(examples["question"], examples["answer"]):
                text = f"### Question: {q}\n### Answer: {a}"
                texts.append(text)
        elif "instruction" in examples and "response" in examples:
            for inst, resp in zip(examples["instruction"], examples["response"]):
                text = f"### Instruction: {inst}\n### Response: {resp}"
                texts.append(text)
        elif "instruction" in examples and "output" in examples:
            for inst, out in zip(examples["instruction"], examples["output"]):
                text = f"### Instruction: {inst}\n### Response: {out}"
                texts.append(text)
        elif "text" in examples:
            texts = examples["text"]
        else:
            # Default: try to find any instruction/response pattern
            keys = list(examples.keys())
            if len(keys) >= 2:
                # Use first two columns as instruction-response
                for item1, item2 in zip(examples[keys[0]], examples[keys[1]]):
                    text = f"### Input: {item1}\n### Output: {item2}"
                    texts.append(text)
            else:
                texts = [str(example) for example in examples[keys[0]]]
        
        return {"text": texts}
    
    # Format the dataset
    formatted_dataset = dataset.map(
        format_instruction,
        batched=True,
        desc="Formatting dataset"
    )
    
    def tokenize_function(examples):
        # Tokenize the texts
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None
        )
        return result
    
    # Tokenize dataset
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    # Create validation split if not exists
    if "validation" not in tokenized_dataset and "test" not in tokenized_dataset:
        print("Creating train/validation split (90/10)...")
        split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
        tokenized_dataset = split_dataset
        # Rename 'test' to 'validation' for consistency
        tokenized_dataset["validation"] = tokenized_dataset.pop("test")
    elif "test" in tokenized_dataset and "validation" not in tokenized_dataset:
        # Rename test to validation
        tokenized_dataset["validation"] = tokenized_dataset.pop("test")
    
    return tokenized_dataset


def create_lora_model(model_name, lora_config, use_4bit=False, use_8bit=False):
    """Load base model and apply LoRA"""
    print(f"Loading base model: {model_name}")
    
    # Configure quantization if requested
    quantization_config = None
    if use_4bit:
        print("Using 4-bit quantization (QLoRA)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    
    # Load model
    model_kwargs = {
        "quantization_config": quantization_config,
        "device_map": "auto" if use_4bit or use_8bit else None,
        "trust_remote_code": True
    }
    
    # Use 'dtype' instead of deprecated 'torch_dtype'
    if use_4bit or use_8bit:
        model_kwargs["dtype"] = torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Prepare model for k-bit training if using quantization
    if use_4bit or use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    print("Applying LoRA configuration")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def main():
    args = parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/{args.model_name.split('/')[-1]}_lora_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("LoRA Fine-tuning Configuration:")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Prepare dataset
    tokenized_dataset = prepare_dataset(
        args.dataset_name,
        args.dataset_config,
        tokenizer,
        args.max_length
    )
    
    # Configure LoRA - detect target modules based on model architecture
    print("\nDetecting target modules for LoRA...")
    if "gpt2" in args.model_name.lower():
        target_modules = ["c_attn", "c_proj"]  # GPT-2 uses different naming
    elif "llama" in args.model_name.lower() or "mistral" in args.model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "opt" in args.model_name.lower():
        target_modules = ["q_proj", "v_proj"]
    else:
        # Default for most models
        target_modules = ["q_proj", "v_proj"]
    
    print(f"Using target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Create model with LoRA
    model = create_lora_model(
        args.model_name,
        lora_config,
        args.use_4bit,
        args.use_8bit
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=args.save_steps,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        fp16=True if (args.use_4bit or args.use_8bit) else False,
        optim="adamw_torch",  # Simplified for CPU/non-quantized
        report_to="none",  # Change to "wandb" or "tensorboard" if needed
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal language modeling
    )
    
    # Determine eval dataset
    eval_dataset_key = "validation" if "validation" in tokenized_dataset else "test"
    eval_dataset = tokenized_dataset.get(eval_dataset_key, None)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    print(f"\nTraining completed! Model saved to: {output_dir}/final")
    print("=" * 60)
    
    # Save training metrics
    metrics_file = f"{output_dir}/training_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("LoRA Fine-tuning Metrics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Dataset: {args.dataset_name}/{args.dataset_config}\n")
        f.write(f"LoRA r: {args.lora_r}\n")
        f.write(f"LoRA alpha: {args.lora_alpha}\n")
        f.write(f"LoRA dropout: {args.lora_dropout}\n")
        f.write(f"Epochs: {args.num_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Quantization: {'4-bit' if args.use_4bit else '8-bit' if args.use_8bit else 'None'}\n")
        f.write("=" * 60 + "\n")
        
        if trainer.state.log_history:
            f.write("\nTraining History:\n")
            for log in trainer.state.log_history:
                f.write(f"{log}\n")
    
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
