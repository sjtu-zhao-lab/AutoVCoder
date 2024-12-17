import os
import torch
import argparse
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator
import transformers
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_model_id(model_name):
    if model_name == "codellama":
        return "codellama/CodeLlama-7b-hf"
    elif model_name == "deepseek":
        return "deepseek-ai/deepseek-coder-6.7b-instruct"
    elif model_name == "codeqwen":
        return "Qwen/CodeQwen1.5-7B"

def main(model_name, dataset_name, DATASET_PATH, CACHE_DIR):
    dataset_path = os.path.join(DATASET_PATH, dataset_name)
    train_dataset = load_dataset('json', data_files=dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model_id = find_model_id(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        cache_dir=CACHE_DIR,
        local_files_only=True,
        load_in_8bit=False,
        trust_remote_code=True
    ).to(device)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        cache_dir=CACHE_DIR,
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    def formatting_func(example):
        text = example["code"]
        return text
    
    def generate_and_tokenize_prompt(prompt):
        result = tokenizer(
            formatting_func(prompt),
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    def print_trainable_parameters(model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
        
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    
    print_trainable_parameters(model)
    print(model)

    # 加速器设置
    accelerator = Accelerator()

    if torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True
        
    model = accelerator.prepare_model(model)

    # 训练设置
    project = "finetune"
    base_model_name = "codellama"
    run_name = f"{model_name}-{project}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    output_dir = f"../../../models/{base_model_name}/{run_name}"

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset['train'],
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=200,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            num_train_epochs=1,
            learning_rate=2.5e-5,
            bf16=True,
            optim="paged_adamw_8bit",
            logging_steps=25,
            logging_dir="./logs",
            save_strategy="steps",
            save_steps=1000,
            do_eval=False,
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model.")
    parser.add_argument('--model_name', type=str, default="codellama", help='Name of the model to use.')
    parser.add_argument('--dataset_name', type=str, default="fine_tune_dataset.jsonl", help='Name of the dataset to use.')
    parser.add_argument('--DATASET_PATH', type=str, default='../../../data/first_round/dataset', help='Path to the dataset directory.')
    parser.add_argument('--CACHE_DIR', type=str, default='../../../downloads/models', help='Path to the cache directory.')

    args = parser.parse_args()

    main(args.model_name, args.dataset_name, args.DATASET_PATH, args.CACHE_DIR)