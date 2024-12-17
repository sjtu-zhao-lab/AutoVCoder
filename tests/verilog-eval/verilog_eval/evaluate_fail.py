import json
import sys

def read_test_jsonl(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            item = json.loads(line)
            data[item.get("task_id")] = {"passed": item.get("passed"), "code": item.get("completion")}
    return data

def read_problem_jsonl(filename, data):
    with open(filename, 'r') as file:
        for line in file:
            item = json.loads(line)
            # print(item.get("task_id"), item.get("task_id") in data, item.get("description"))
            if item.get("task_id") in data:
                data[item.get("task_id")]["problem"] = item.get("detail_description")
    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <jsonl_file>")
        return
    
    input_jsonl = sys.argv[1]
    verilog_jsonl = "../descriptions/VerilogDescription_Human.jsonl"

    # 读取输入的 JSONL 文件
    data = input_data = read_test_jsonl(input_jsonl)

    # print(data)

    # 读取 Verilog 描述的 JSONL 文件
    data = verilog_data = read_problem_jsonl(verilog_jsonl, data)

    # print(input_data)
    # print(data)

    # 找到未通过的任务并输出描述
    cnt = 0
    for task_id in data:
        if data[task_id].get("passed") == False:
            cnt += 1
            print(f"Task {cnt}: {task_id}\nDescription: {data[task_id].get('problem')}\nCode: {data[task_id].get('code')}\n")

if __name__ == "__main__":
    main()
