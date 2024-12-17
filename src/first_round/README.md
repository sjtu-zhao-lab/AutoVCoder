# First Round Fine-tuning

## Build Dataset

Our dataset is directly scraped from Verilog data available on GitHub. The dataset construction process is as follows, with the working directory located at ./build_dataset:
1. Identify Verilog Repositories: The dataset for this project is sourced from repositories active between January 1, 2010, and January 1, 2023. To generate a list of these repositories, run:
```
python find_repos.py
```
This will produce a list of repositories and save it to a CSV file.

2. Clone Repositories: Use the following command to clone the repositories locally:
```
python clone_repos.py
```

3. Process Verilog Files: Locate .v files within each repository and split them into modules for further processing.

## LLM fine-tuning

The current framework supports LoRA fine-tuning on a single GPU. For multi-GPU model fine-tuning, you can refer to frameworks like Pytroch DDP and Megatron-LM. Our framework supports fine-tuning for three models: 

- codellama: codellama/CodeLlama-7b-hf
- deepseek: deepseek-ai/deepseek-coder-6.7b-instruct
- codeqwen: Qwen/CodeQwen1.5-7B

To fine-tune the model, run the following command:
```
python train.py --model_name {model_name} --dataset_name {dataset_name} --CACHE_DIR {CACHE_DIR} --DATASET_PATH {DATASET_PATH}
```

For example, to fine-tune the codellama model, run the following command:
```
python train.py --model_name codellama --dataset_name fine_tune_dataset.jsonl --CACHE_DIR ../../../downloads/models --DATASET_PATH ../../../data/first_round/dataset
```

To save the fine-tuned codellama model, run the following command:
```
python save_lora_model.py --model_name {model_name} --peft_model_id {peft_model_id} --output_path {output_path} --CACHE_DIR {CACHE_DIR}
```



