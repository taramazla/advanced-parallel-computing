"""
Script to prepare datasets for LoRA fine-tuning
Supports HuggingFace datasets and local files (JSON, CSV, TXT)
"""

import json
import csv
import argparse
from datasets import Dataset, DatasetDict, load_dataset
from typing import List, Dict
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning")
    parser.add_argument("--dataset_name", type=str, default="bitext/Bitext-customer-support-llm-chatbot-training-dataset",
                        help="HuggingFace dataset name (e.g., bitext/Bitext-customer-support-llm-chatbot-training-dataset)")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration (optional)")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Local input file path (JSON, CSV, or TXT). If provided, will use local file instead of HuggingFace")
    parser.add_argument("--output_dir", type=str, default="./custom_dataset",
                        help="Output directory for processed dataset")
    parser.add_argument("--format", type=str, choices=["json", "csv", "txt", "instruction", "huggingface"],
                        default="huggingface", help="Input format (default: huggingface)")
    parser.add_argument("--text_column", type=str, default="text",
                        help="Column name containing text data (for local files)")
    parser.add_argument("--train_split", type=float, default=0.9,
                        help="Proportion of data for training")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use")
    return parser.parse_args()


def load_huggingface_dataset(dataset_name: str, dataset_config: str, max_samples: int = None) -> DatasetDict:
    """Load dataset from HuggingFace"""
    if dataset_config:
        print(f"Loading dataset from HuggingFace: {dataset_name}/{dataset_config}")
        try:
            dataset = load_dataset(dataset_name, dataset_config, trust_remote_code=True)
        except:
            print(f"Failed to load with config, trying without config...")
            dataset = load_dataset(dataset_name, trust_remote_code=True)
    else:
        print(f"Loading dataset from HuggingFace: {dataset_name}")
        dataset = load_dataset(dataset_name, trust_remote_code=True)
    
    # Limit samples if specified
    if max_samples:
        if "train" in dataset:
            dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
        if "validation" in dataset:
            val_samples = min(max_samples // 10, len(dataset["validation"]))
            dataset["validation"] = dataset["validation"].select(range(val_samples))
        elif "test" in dataset:
            val_samples = min(max_samples // 10, len(dataset["test"]))
            dataset["test"] = dataset["test"].select(range(val_samples))
    
    print(f"Loaded {len(dataset['train']) if 'train' in dataset else 0} training samples")
    print(f"Loaded {len(dataset['validation']) if 'validation' in dataset else len(dataset.get('test', []))} validation/test samples")
    
    return dataset


def load_json_data(file_path: str, text_column: str, max_samples: int = None) -> List[Dict]:
    """Load data from JSON file"""
    print(f"Loading JSON data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict) and text_column in data:
        # Convert dict of lists to list of dicts
        samples = [{text_column: text} for text in data[text_column]]
    else:
        raise ValueError("Unsupported JSON structure")
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Loaded {len(samples)} samples")
    return samples


def load_csv_data(file_path: str, text_column: str, max_samples: int = None) -> List[Dict]:
    """Load data from CSV file"""
    print(f"Loading CSV data from {file_path}")
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            if text_column in row:
                samples.append({text_column: row[text_column]})
    
    print(f"Loaded {len(samples)} samples")
    return samples


def load_txt_data(file_path: str, text_column: str, max_samples: int = None) -> List[Dict]:
    """Load data from TXT file (one sample per line)"""
    print(f"Loading TXT data from {file_path}")
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            line = line.strip()
            if line:
                samples.append({text_column: line})
    
    print(f"Loaded {len(samples)} samples")
    return samples


def load_instruction_data(file_path: str, max_samples: int = None) -> List[Dict]:
    """Load instruction-tuning format data (instruction, input, output)"""
    print(f"Loading instruction data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break
        
        # Format as instruction-following prompt
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        
        samples.append({"text": prompt})
    
    print(f"Loaded {len(samples)} instruction samples")
    return samples


def create_dataset_split(samples: List[Dict], train_split: float) -> DatasetDict:
    """Split data into train and validation sets"""
    dataset = Dataset.from_list(samples)
    
    split_idx = int(len(samples) * train_split)
    train_dataset = dataset.select(range(split_idx))
    val_dataset = dataset.select(range(split_idx, len(samples)))
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })
    
    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    return dataset_dict


def main():
    args = parse_args()
    
    # Load data based on format
    if args.format == "huggingface" or args.input_file is None:
        # Load from HuggingFace
        print("Using HuggingFace dataset...")
        dataset_dict = load_huggingface_dataset(args.dataset_name, args.dataset_config, args.max_samples)
        
        # If dataset doesn't have validation split, create one
        if "validation" not in dataset_dict and "train" in dataset_dict:
            print("Creating train/validation split...")
            split_dataset = dataset_dict["train"].train_test_split(
                test_size=1-args.train_split,
                seed=42
            )
            dataset_dict = DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            })
    else:
        # Load from local file
        print("Using local file...")
        if args.format == "json":
            samples = load_json_data(args.input_file, args.text_column, args.max_samples)
        elif args.format == "csv":
            samples = load_csv_data(args.input_file, args.text_column, args.max_samples)
        elif args.format == "txt":
            samples = load_txt_data(args.input_file, args.text_column, args.max_samples)
        elif args.format == "instruction":
            samples = load_instruction_data(args.input_file, args.max_samples)
        else:
            raise ValueError(f"Unsupported format: {args.format}")
        
        # Create train/val split
        dataset_dict = create_dataset_split(samples, args.train_split)
    
    # Save dataset
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)
    print(f"\nDataset saved to: {args.output_dir}")
    
    # Save statistics
    stats_file = os.path.join(args.output_dir, "dataset_stats.txt")
    with open(stats_file, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("=" * 60 + "\n")
        if args.format == "huggingface":
            f.write(f"Dataset: {args.dataset_name}\n")
            f.write(f"Config: {args.dataset_config}\n")
        else:
            f.write(f"Source file: {args.input_file}\n")
            f.write(f"Format: {args.format}\n")
        f.write(f"Training samples: {len(dataset_dict['train'])}\n")
        f.write(f"Validation samples: {len(dataset_dict['validation'])}\n")
        f.write(f"Train/Val split: {args.train_split}/{1-args.train_split}\n")
        f.write("=" * 60 + "\n")
        
        # Sample examples
        f.write("\nSample Training Examples:\n")
        for i, example in enumerate(dataset_dict['train'].select(range(min(3, len(dataset_dict['train']))))):
            f.write(f"\nExample {i+1}:\n")
            f.write(f"{example}\n")
            f.write("-" * 60 + "\n")
    
    print(f"Statistics saved to: {stats_file}")
    print("\nDataset preparation completed!")


if __name__ == "__main__":
    main()
