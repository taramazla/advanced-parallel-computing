"""
Inference script for LoRA fine-tuned models
Supports text generation and interactive chat
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with LoRA fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned LoRA model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model name (auto-detected if not provided)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Input prompt for generation")
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive mode")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling top-p")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty")
    parser.add_argument("--do_sample", action="store_true",
                        help="Use sampling instead of greedy decoding")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, base_model=None):
    """Load LoRA model and tokenizer"""
    print("Loading model and tokenizer...")
    
    # Load LoRA config to get base model if not provided
    if base_model is None:
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model = peft_config.base_model_name_or_path
        print(f"Detected base model: {base_model}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if base_model is None else base_model,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load base model
    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    print(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    # Merge LoRA weights for faster inference (optional)
    # model = model.merge_and_unload()
    
    model.eval()
    print("Model loaded successfully!")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, args):
    """Generate text from a prompt"""
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def interactive_mode(model, tokenizer, args):
    """Interactive chat mode"""
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")
    
    while True:
        # Get user input
        prompt = input("You: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break
        
        if not prompt:
            continue
        
        # Generate response
        print("\nGenerating response...\n")
        response = generate_text(model, tokenizer, prompt, args)
        
        print(f"Assistant: {response}\n")
        print("-" * 60 + "\n")


def batch_inference(model, tokenizer, prompts_file, output_file, args):
    """Process multiple prompts from a file"""
    print(f"Processing prompts from: {prompts_file}")
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing prompt {i}/{len(prompts)}")
        print(f"Prompt: {prompt[:100]}...")
        
        response = generate_text(model, tokenizer, prompt, args)
        results.append({
            "prompt": prompt,
            "response": response
        })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")


def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(model, tokenizer, args)
    
    elif args.prompt:
        # Single prompt generation
        print("\n" + "=" * 60)
        print("Input Prompt:")
        print("=" * 60)
        print(args.prompt)
        print("\n" + "=" * 60)
        print("Generated Output:")
        print("=" * 60)
        
        response = generate_text(model, tokenizer, args.prompt, args)
        print(response)
        print("=" * 60 + "\n")
    
    else:
        print("Please provide either --prompt or --interactive flag")
        print("Example: python inference.py --model_path ./lora_outputs/final --prompt 'Your prompt here'")
        print("Example: python inference.py --model_path ./lora_outputs/final --interactive")


if __name__ == "__main__":
    main()
