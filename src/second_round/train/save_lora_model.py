import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def find_model_id(model_name):
    if model_name == "codellama":
        return "codellama/CodeLlama-7b-hf"
    elif model_name == "deepseek":
        return "deepseek-ai/deepseek-coder-6.7b-instruct"
    elif model_name == "codeqwen":
        return "Qwen/CodeQwen1.5-7B" 

def apply_lora(model_name, peft_model_id, output_path, CACHE_DIR):
    print(f"Loading the base model from {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        load_in_8bit=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=CACHE_DIR
    )
    
    print(f"Loading the LoRA adapter from {peft_model_id}")
    model = PeftModel.from_pretrained(model, peft_model_id)
 
    print("Applying the LoRA")
    model = model.merge_and_unload()
 
    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply LoRA to a language model.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to use.')
    parser.add_argument('--peft_model_id', type=str, required=True, help='Path to the LoRA adapter checkpoint.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the modified model.')
    parser.add_argument('--CACHE_DIR', type=str, default='../../../downloads/models', help='Path to the cache directory.')

    args = parser.parse_args()

    apply_lora(args.model_name, args.peft_model_id, args.output_path, args.CACHE_DIR)