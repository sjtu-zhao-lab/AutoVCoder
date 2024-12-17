import json
import torch
import torch.nn as nn
from tqdm import tqdm

# 定义MLP模型
class MLPModel(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=[1024, 512, 256], output_dim=1):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_dim)
        )

    def forward(self, x):
        return self.layers(x)

def build_dataset():
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPModel().to(device)
    model.load_state_dict(torch.load('../../../models/code_scorer.pth'))
    model.eval()

    # 加载嵌入
    embeddings_pt = torch.load('../../../data/first_round/embeddings.pt')
    embeddings = {code[:20]: emb for code, emb in zip(embeddings_pt["codes"], embeddings_pt["embeddings"])}

    # 读取 origin.jsonl 并进行评分
    with open('../../../data/first_round/origin.jsonl', 'r') as infile:
        lines = infile.readlines()  # 读取所有行以计算总行数

    with open('../../../data/first_round/origin_scored.jsonl', 'w') as outfile, \
         open('../../../data/first_round/dataset/fine_tune_dataset.jsonl', 'w') as fine_tune_file:
        
        for line in tqdm(lines, total=len(lines), desc="Processing"):
            data = json.loads(line)
            code_key = data['code'][:20]  # 只取前20个字符作为键
            emb = embeddings.get(code_key)
            if emb is None:
                continue
            
            # 将嵌入转换为张量并移动到设备
            emb_tensor = torch.tensor(emb, dtype=torch.float32).to(device)
            
            # 计算分数
            with torch.no_grad():
                score = model(emb_tensor.unsqueeze(0)).item()
              
            result = {"code": data['code'], "score": score}
            outfile.write(json.dumps(result) + '\n')
            
            # 将分数大于6.5的写入'../../../data/first_round/dataset/fine_tune_dataset.jsonl'
            if score > 6.5:
                fine_tune_file.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    build_dataset()