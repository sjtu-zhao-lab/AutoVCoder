import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import gc
import re
import csv
import json
import torch
import tempfile
import transformers
import subprocess
import argparse

from tqdm import tqdm
from peft import PeftModel, PeftConfig
from pathlib import Path
from pyverilog.vparser.parser import parse
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map='auto',
        cache_dir='../model', 
        local_files_only=True
    ).to('cuda')
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        cache_dir='../model'
    )
    return model, tokenizer

def load_lora_model(model_name, round):
    if round == 'first':
        if 'deepseek' in model_name:
            model_path = "deepseek-ai/deepseek-coder-6.7b-instruct"
            peft_model_id = "../models/deepseek/deepseek-finetune-2024-11-26-10-55/checkpoint-84453"
        elif 'codellama' in model_name:
            model_path = "codellama/CodeLlama-7b-hf"
            peft_model_id = "../models/codellama/codellama-finetune-2024-11-12-15-11/checkpoint-84453"
        elif 'codeqwen' in model_name:
            model_path = "Qwen/CodeQwen1.5-7B"
            peft_model_id = "../models/codeqwen/codeqwen-finetune-2024-11-26-10-46/checkpoint-84453"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            return_dict=True, 
            device_map='auto',
            cache_dir='../downloads/models',
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir='../downloads/models',
        )
    else:
        CACHE_DIR = '../models/first_round_models/'
        model_path = os.path.join(CACHE_DIR, model_name)
        tokenizer_path = os.path.join(CACHE_DIR, model_name)
        
        if 'deepseek' in model_name:
            peft_model_id = "../models/deepseek/deepseek-finetune-2024-12-02-11-03/checkpoint-33828"
        elif 'codellama' in model_name:
            peft_model_id = "../models/codellama/codellama-finetune-2024-12-02-11-03/checkpoint-33828"
        elif 'codeqwen' in model_name:
            peft_model_id = "../models/codeqwen/codeqwen-finetune-2024-12-02-11-01/checkpoint-33828"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            return_dict=True, 
            device_map='auto',
            cache_dir=CACHE_DIR,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=CACHE_DIR,
        )
    
    model = PeftModel.from_pretrained(model, peft_model_id)
    model = model.to('cuda')

    return model, tokenizer

def verilog_generate(prompts, model, tokenizer):
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        add_special_tokens=True, 
        return_attention_mask=True
    )
    input_ids = inputs["input_ids"].to('cuda')
    attention_masks = inputs['attention_mask'].to('cuda')

    generation_config = transformers.GenerationConfig(
        do_sample = False,
        max_new_tokens = 500, 
        # tempureture = 0.95
    )

    # print('input_ids', input_ids)
    # print('attention_mask', attention_masks)
    
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id
        )

    output_texts = []
    for i in range(len(input_ids)):
        output_text = tokenizer.decode(generation_output[i][len(input_ids[i]):].cuda(), skip_special_tokens=True).strip()
        output_texts.append(output_text)
    # print(output_texts)
    return output_texts

def syntactic_analyse(verilog_code):
    # 创建一个临时文件
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.write(verilog_code)
        temp_file_name = temp_file.name

    # 解析这个临时文件
    try:
        ast, directives = parse([temp_file_name])
        return True
    except Exception as e:
        return False

def functionality_analyse(task, verilog_code):
    testbench_path = task / 'testbench.v'
    # testbench_path = 'test_case/accu/testbench.v'

    try:
        with open(testbench_path, 'r') as file:
            testbench_code = file.read()

        modified_testbench = verilog_code + "\n" + testbench_code

        # 将修改后的测试台代码写入文件
        with open('testbench.v', 'w') as file:
            file.write(modified_testbench)

        # 使用iverilog编译testbench，生成vvp可执行文件
        # 运行vvp仿真，并捕获输出
        try:
            compile_command = f"iverilog -o simulation testbench.v"
            subprocess.run(compile_command, shell=True, check=True)
            simulation_command = f"vvp simulation -lxt2"
            result = subprocess.run(simulation_command, shell=True, check=True, stdout=subprocess.PIPE, text=True, timeout=20)
        except:
            return False
 
        # 分析输出
        if "Your Design Passed" in result.stdout:
            return True
        else:
            return False
    finally:
        # 删除testbench文件
        os.remove('testbench.v')
        if os.path.exists('simulation'):
            os.remove('simulation')
        pass
    
def fetch_task_prompt(task):
    file_path = task / 'design_description.txt'
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt

def load_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def auto_test_rtllm(model_name, model, tokenizer, round):
    # 遍历 base_path 下的所有子文件夹
    base_path = Path('rtllm')
    csv_file_path = f"{model_name}.csv"

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Task', 'Prompt', 'Verilog Code', 'Syntax Check', 'Functionality Check'])

        cnt, syntax_pass_cnt, function_pass_cnt = 0, 0, 0

        tasks, prompts = [], []
        for task in base_path.iterdir():
            if task.is_dir():
                prompt = fetch_task_prompt(task)
                tasks.append(task)
                prompts.append(prompt)
        
        verilog_codes = []
        for idx in tqdm(range(0, len(prompts) // 8 + 1)):
            if idx*8+7 < len(prompts):
                prompts_batch = prompts[idx*8:idx*8+8]
            else:
                prompts_batch = prompts[idx*8:]
            verilog_codes_batch = verilog_generate(prompts_batch, model, tokenizer)
            print(verilog_codes_batch)
            verilog_codes.extend(verilog_codes_batch)

        for i, verilog_code in enumerate(verilog_codes):
            task = tasks[i]
            prompt = prompts[i]
            verilog_code = verilog_code.replace("Module", "module")

            phrase = "Give me the complete code."

            # 在verilog_code找到"Give me the complete code."这句话，然后把前面的全部删掉
            index = verilog_code.find(phrase)
            if index != -1:
                verilog_code = verilog_code[index + len(phrase):]

            pattern = re.compile(r'module\s+.*?endmodule', re.DOTALL)  # re.DOTALL 让 . 匹配包括换行在内的任意字符
            mat = pattern.search(verilog_code)

            if mat:
                verilog_code = mat.group()
                # print(f"Task: {task}, Verilog code: {verilog_code}")       
                syntactic_flag = syntactic_analyse(verilog_code)
                if syntactic_flag:
                    functionality_flag = functionality_analyse(task, verilog_code)
                else:
                    functionality_flag = False
            else:
                syntactic_flag = False
                functionality_flag = False

            cnt += 1
            if syntactic_flag:
                syntax_pass_cnt += 1
            if functionality_flag:
                function_pass_cnt += 1

            print(f"Task: {task}, Syntax: {syntactic_flag}, Function: {functionality_flag}\n")
            writer.writerow([task.name, prompt, verilog_code, syntactic_flag, functionality_flag])

        print(f"Syntax Pass Rate: {syntax_pass_cnt/cnt*100}%, Function Pass Rate: {function_pass_cnt/cnt*100}%")

def auto_test_verilogeval(model_name, model, tokenizer, round, mode):
    problems = {}
    result_file_name = f'results/{model_name}_verilog-eval_{mode}_{round}.jsonl'

    descriptions = load_jsonl(f'verilog-eval/descriptions/VerilogDescription_{mode}.jsonl')
    data = load_jsonl(f'verilog-eval/data/VerilogEval_{mode}.jsonl')

    for desc in descriptions:
        task_id = desc.get('task_id')
        problems[task_id] = {'problem': desc.get('detail_description')}

    for datum in data:
        task_id = datum.get('task_id')
        if task_id in problems:
            problems[task_id]['module'] = datum.get('prompt')

    tasks, prompts = [], []
    for task_id, content in problems.items():
        prompt = f"{content['problem']}{content['module']}. Please give me the whole Verilog code: "
        tasks.append(task_id)
        prompts.append(prompt)

    batch_size = 16
    with open(result_file_name, 'w') as f:
        for idx in tqdm(range(0, len(prompts), batch_size)):
            prompts_batch = prompts[idx:idx + batch_size]
            verilog_codes_batch = verilog_generate(prompts_batch, model, tokenizer)
            
            for i, verilog_code in enumerate(verilog_codes_batch):
                task_id = tasks[idx + i]
                pattern = re.compile(r'module\s+.*?endmodule', re.DOTALL)
                mat = pattern.search(verilog_code)
                if mat:
                    verilog_code = mat.group()
                new_data = {
                    "task_id": task_id,
                    "completion": verilog_code
                }
                f.write(json.dumps(new_data) + '\n')
    try:
        result = subprocess.run(
            f"cd verilog-eval && evaluate_functional_correctness ../{result_file_name} --problem_file data/VerilogEval_{mode}.jsonl",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            text=True
        )
        with open(f'results/{model_name}_verilog-eval_{mode}_{round}.log', 'w') as f:
            f.write(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while evaluating functional correctness: {e}")
        
def main(model_name, benchmark, round, mode=None):
    model, tokenizer = load_lora_model(model_name, round)

    test_functions = {
        'rtllm': auto_test_rtllm,
        'verilogeval': auto_test_verilogeval
    }
    try:
        short_model_name = model_name.split('/')[1]
    except:
        short_model_name = model_name
    test_function = test_functions.get(benchmark, auto_test_verilogeval)  # 默认选择 'verilogeval'
    if test_function:
        if benchmark == 'verilogeval':
            test_function(short_model_name, model, tokenizer, round, mode)
        else:
            test_function(short_model_name, model, tokenizer, round)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model benchmark tests.")
    parser.add_argument('--model', type=str, default="codellama", help='The model name to use.')
    parser.add_argument('--benchmark', type=str, default="verilogeval", choices=['rtllm', 'verilogeval'], help='The benchmark type to run.')
    parser.add_argument('--mode', type=str, choices=['Human', 'Machine'], help='The mode for verilogeval benchmark.')
    parser.add_argument('--round', type=str, default="first", choices=['first', 'second'], help='The round of the model.')
    args = parser.parse_args()

    if args.benchmark == 'verilogeval' and not args.mode:
        parser.error("--mode is required when benchmark is 'verilogeval'")

    main(args.model, args.benchmark, args.round, args.mode)

    