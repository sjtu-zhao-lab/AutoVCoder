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

4.  After evaluating the code with the **code scorer**, save the code with scores exceeding 6.5 to data/first_round/dataset/fine_tune_dataset.jsonl. To execute this process, run:
```
python build_dataset.py
```

## Code Scorer

![code_scorer](/pics/2.png)


In this process, we utilize the bge-m3 sentence embedding model from FlagEmbedding as the embedding model for our code scorer.The steps to construct the scorer are as follows, with the working directory located at ./code_scorer:

1. The processed raw data is stored in data/first_round/origin.jsonl. We randomly select 15,000 entries from this dataset and save them to data/first_round/score_by_gpt.jsonl. To perform this selection and scoring, run:
```
python code_mark_by_gpt.py
```
This will generate scores for the 15,000 entries and save them in data/first_round/score_by_gpt.jsonl.
2. Encode all code embeddings using the bge-m3 model and save the results to data/first_round/embeddings.pt. Pre-encoding the data helps reduce the time required for training and inference. To obtain the encoded results, run:
```
python code_encode.py
```
Concatenate the encoded vectors with the scored data to create a dataset for training the MLP. To build the dataset and train the code scorer model, run:
```
python build_dataset.py
python train_model.py
```
This will produce the trained code scorer model.

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



