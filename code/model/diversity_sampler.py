import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from sklearn.cluster import MiniBatchKMeans
import sys
import shutil

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def random_file_prefix(prefix_name='tmp'):
    """生成随机文件名"""
    random_bytes = os.urandom(8)
    random_str = random_bytes.hex()
    temp_filename = f"{prefix_name}.{random_str}.mlir"
    return temp_filename

def put_file_content(path, content):
    """写入文件内容"""
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
        f.flush()

        

class VectorDataset(Dataset):
    """用于批量计算向量的数据集"""
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length"
        )
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        return inputs


class DiversitySampler:
    """基于神经网络向量的多样性采样器"""
    
    def __init__(self, model_path="./codegen-2B", checkpoint_path="checkpointSearch/", 
                 batch_size=32):
        """
        初始化采样器
        
        Args:
            model_path: 预训练模型路径
            checkpoint_path: LoRA checkpoint路径
            batch_size: 批量计算大小
        """
        # Initialize the Accelerator
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.device = self.accelerator.device
        self.batch_size = batch_size
        
        # 多GPU信息
        self.is_main_process = self.accelerator.is_main_process
        self.num_processes = self.accelerator.num_processes
        
        if self.is_main_process:
            print(f"[Info] Using {self.num_processes} GPUs for distributed inference")
            print(f"[Info] Main process device: {self.device}")
        
        print("[Info] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("[Info] Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # 应用LoRA配置
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", 
            inference_mode=False,  # 参考infer2.py设置为False
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, peft_config)

        # 加载checkpoint（在prepare之前）
        self._load_checkpoint(checkpoint_path)
        
        # 创建optimizer用于DeepSpeed
        from torch import optim
        optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
        
        # 创建一个dummy dataloader用于满足DeepSpeed要求
        dummy_dataset = torch.utils.data.TensorDataset(torch.zeros(1, 1))
        dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=self.batch_size)
        
        # Prepare model with accelerator (包含optimizer和dataloader)
        self.model, optimizer, dummy_loader = self.accelerator.prepare(self.model, optimizer, dummy_loader)
        self.model.eval()
        
        # 缓存已计算的向量
        self.vector_cache = {}
        
    def _load_checkpoint(self, checkpoint_path):
        """加载模型checkpoint"""
        ckpt_file = os.path.join(checkpoint_path, 'best_model.ckpt')
        if os.path.exists(ckpt_file):
            print(f"[Info] Loading checkpoint from {ckpt_file}")
            self.model.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
        else:
            print(f"[Warning] Checkpoint not found at {ckpt_file}, using base model")
            print(f"[Warning] Make sure the checkpoint file exists at: {ckpt_file}")
    
    @torch.no_grad()
    def compute_vectors_batch(self, texts):
        """
        批量计算文本的向量表示（分布式版本）
        
        使用 accelerator.prepare 自动分发数据到多卡，然后 gather 收集结果。
        
        Args:
            texts: 文本列表
            
        Returns:
            numpy array of shape (len(texts), hidden_size)
        """
        # 过滤已缓存的
        uncached_indices = []
        uncached_texts = []
        for i, text in enumerate(texts):
            if text not in self.vector_cache:
                uncached_indices.append(i)
                uncached_texts.append(text)
        
        # 计算未缓存的向量
        if uncached_texts:
            dataset = VectorDataset(uncached_texts, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            # 关键：使用 accelerator.prepare 分发数据到多卡
            dataloader = self.accelerator.prepare(dataloader)
            
            local_vectors = []
            local_indices = []  # 记录本进程处理的原始索引
            
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing vectors", 
                                                    leave=False, disable=not self.is_main_process)):
                input_ids = batch['input_ids'].to(self.accelerator.device)
                attention_mask = batch['attention_mask'].to(self.accelerator.device)
                
                # 获取模型的隐藏状态
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # 使用最后一层的平均池化作为向量表示
                hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
                
                # 使用attention mask进行平均池化
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask  # (batch, hidden_size)
                
                local_vectors.append(pooled)
            
            # 同步所有进程
            self.accelerator.wait_for_everyone()
            
            if local_vectors:
                # 合并本进程的所有向量
                local_vectors_tensor = torch.cat(local_vectors, dim=0)
                
                # 使用 gather 收集所有进程的结果
                all_vectors_tensor = self.accelerator.gather(local_vectors_tensor)
                
                # 截取有效部分（去除因 padding 产生的多余数据）
                vectors = all_vectors_tensor[:len(uncached_texts)].cpu().numpy()
            else:
                vectors = np.zeros((len(uncached_texts), self._get_hidden_size()), dtype=np.float32)
            
            # 更新缓存（所有进程都更新，因为 gather 后所有进程都有完整结果）
            for i, text in enumerate(uncached_texts):
                self.vector_cache[text] = vectors[i]
        
        # 组装结果
        result = np.zeros((len(texts), self._get_hidden_size()), dtype=np.float32)
        for i, text in enumerate(texts):
            result[i] = self.vector_cache[text]
        
        return result
    
    def _get_hidden_size(self):
        """获取隐藏层大小（兼容DeepSpeed包装的模型）"""
        # DeepSpeed包装后，需要通过module访问原始模型
        model = self.accelerator.unwrap_model(self.model)
        config = model.config
        if isinstance(config, dict):
            return config.get('hidden_size', config.get('n_embd', 2560))
        return config.hidden_size
    
    def compute_diversity_score(self, candidate_vectors, selected_vectors, method='min_distance'):
        """
        计算候选向量相对于已选集合的多样性分数
        
        Args:
            candidate_vectors: 候选向量 (n_candidates, hidden_size)
            selected_vectors: 已选向量 (n_selected, hidden_size)
            method: 计算方法
                - 'min_distance': 与最近邻的距离（越大越多样）
                - 'avg_distance': 与所有已选的平均距离
                - 'max_similarity': 与最相似的相似度（越小越多样）
                
        Returns:
            diversity_scores: (n_candidates,) 多样性分数，越大越好
        """
        # 归一化向量
        candidate_norm = candidate_vectors / (np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-9)
        selected_norm = selected_vectors / (np.linalg.norm(selected_vectors, axis=1, keepdims=True) + 1e-9)
        
        # 计算余弦相似度矩阵 (n_candidates, n_selected)
        similarity_matrix = np.dot(candidate_norm, selected_norm.T)
        
        if method == 'min_distance':
            # 最大相似度 -> 最小距离的反面
            max_similarity = np.max(similarity_matrix, axis=1)
            diversity_scores = 1 - max_similarity  # 转换为多样性分数
            
        elif method == 'avg_distance':
            # 平均相似度
            avg_similarity = np.mean(similarity_matrix, axis=1)
            diversity_scores = 1 - avg_similarity
            
        elif method == 'max_similarity':
            # 直接返回负的最大相似度
            max_similarity = np.max(similarity_matrix, axis=1)
            diversity_scores = -max_similarity
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return diversity_scores
    
    def sample(self, all_texts, target_count=100000, 
               initial_count=2000, candidate_count=10000, 
               select_ratio=0.2, diversity_method='min_distance',
               save_file=None, save_path=None, save_interval=10000, seed=42):
        """
        执行多样性采样
        
        Args:
            all_texts: 所有可用的文本列表
            target_count: 目标采样数量
            initial_count: 初始随机采样数量
            candidate_count: 每轮候选数量
            select_ratio: 每轮选择比例
            diversity_method: 多样性计算方法
            save_path: 中间结果保存路径
            save_interval: 保存间隔
            seed: 随机种子，确保所有进程使用相同的随机序列
            
        Returns:
            selected_texts: 选中的文本列表
        """
        # 设置随机种子，确保所有进程的随机采样一致
        random.seed(seed)
        np.random.seed(seed)
        
        if self.is_main_process:
            print(f"\n[Info] Starting diversity sampling")
            print(f"[Info] Total available: {len(all_texts)}")
            print(f"[Info] Target count: {target_count}")
            print(f"[Info] Initial count: {initial_count}")
            print(f"[Info] Candidate count per round: {candidate_count}")
            print(f"[Info] Select ratio per round: {select_ratio}")
        
        # 确保目标数量不超过总数
        target_count = min(target_count, len(all_texts))
        
        # 创建索引集合
        all_indices = set(range(len(all_texts)))
        selected_indices = set()
        
        # Step 1: 初始随机采样
        if self.is_main_process:
            print(f"\n[Step 1] Initial random sampling: {initial_count} samples")
        initial_indices = random.sample(list(all_indices), min(initial_count, len(all_indices)))
        selected_indices.update(initial_indices)
        all_indices -= selected_indices
        
        # 计算初始样本的向量
        initial_texts = [all_texts[i] for i in initial_indices]
        selected_vectors = self.compute_vectors_batch(initial_texts)
        
        if self.is_main_process:
            print(f"[Info] Initial sampling done. Selected: {len(selected_indices)}")
        
        # Step 2: 迭代选择
        round_num = 0
        while len(selected_indices) < target_count and len(all_indices) > 0:
            round_num += 1
            
            # 计算本轮需要选择的数量
            remaining = target_count - len(selected_indices)
            select_count = min(int(candidate_count * select_ratio), remaining)
            
            if select_count <= 0:
                break
            
            # 随机选取候选
            actual_candidate_count = min(candidate_count, len(all_indices))
            candidate_indices = random.sample(list(all_indices), actual_candidate_count)
            candidate_texts = [all_texts[i] for i in candidate_indices]
            
            if self.is_main_process:
                print(f"\n[Round {round_num}] Candidates: {len(candidate_indices)}, "
                      f"To select: {select_count}, "
                      f"Current selected: {len(selected_indices)}")
            
            # 计算候选向量
            candidate_vectors = self.compute_vectors_batch(candidate_texts)
            
            # 计算多样性分数
            diversity_scores = self.compute_diversity_score(
                candidate_vectors, selected_vectors, method=diversity_method
            )
            
            # 选择多样性最高的
            top_indices = np.argsort(diversity_scores)[-select_count:]
            
            # 更新已选集合
            new_selected_indices = [candidate_indices[i] for i in top_indices]
            new_selected_vectors = candidate_vectors[top_indices]
            
            selected_indices.update(new_selected_indices)
            all_indices -= set(new_selected_indices)
            
            # 更新已选向量（增量更新）
            selected_vectors = np.concatenate([selected_vectors, new_selected_vectors], axis=0)
            
            if self.is_main_process:
                print(f"[Round {round_num}] Added {len(new_selected_indices)} samples. "
                      f"Total selected: {len(selected_indices)}")

        # 最终保存
        if save_file:
            self._save_to_json(all_texts, selected_indices, save_file)
        
        # 返回选中的文本
        selected_texts = [all_texts[i] for i in selected_indices]
        
        if self.is_main_process:
            print(f"\n[Info] Sampling completed!")
            print(f"[Info] Total selected: {len(selected_texts)}")
        
        return selected_texts
    
    def _save_to_json(self, all_texts, selected_indices, save_file):
        """
        保存 all_texts 和 selected_indices 到 JSON 文件
        
        Args:
            all_texts: 所有文本列表
            selected_indices: 选中的索引集合或列表
            output_path: 输出JSON文件路径
        """
        # 确保目录存在
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        selected_texts = [all_texts[i] for i in selected_indices]
        
        # 保存到JSON文件
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(selected_texts, f, ensure_ascii=False, indent=2)
        
   
    def _save_progress(self, all_texts, selected_indices, save_dir, tag):
        """保存中间进度到单个文件"""
        os.makedirs(save_dir, exist_ok=True)
        
        selected_texts = [all_texts[i] for i in selected_indices]
        
        # 保存为单个文件，每个程序一个文件
        saved_count = 0
        for text in tqdm(selected_texts, desc=f"Saving {tag}"):
            file_name = os.path.join(save_dir, random_file_prefix())
            try:
                # 如果是JSON字符串，保持原样；否则直接保存
                content = text
                put_file_content(file_name, content)
                saved_count += 1
            except Exception as e:
                print(f"[Warning] Failed to save file: {e}")
        
        print(f"[Info] Progress saved: {saved_count} files to {save_dir}")


def load_programs_from_directory(directory):
    """从目录加载所有程序"""
    programs = []
    path = Path(directory)
    
    for file_path in tqdm(list(path.rglob('*')), desc="Loading programs"):
        if file_path.is_file():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 尝试解析JSON
                    if content.startswith('"'):
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            pass
                    programs.append(content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    return programs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Diversity-based sampling using neural network vectors')
    parser.add_argument('--seed_dir', type=str, help='Directory containing seed programs')
    parser.add_argument('--input_json', type=str, help='Input JSON file containing programs')
    parser.add_argument('--output_dir', type=str, help='Output directory for sampled programs')
    parser.add_argument('--output_json', type=str, help='Output JSON file for sampled programs')
    parser.add_argument('--target_count', type=int, default=100000, help='Target number of samples')
    parser.add_argument('--initial_count', type=int, default=2000, help='Initial random sample count')
    parser.add_argument('--candidate_count', type=int, default=10000, help='Candidates per round')
    parser.add_argument('--select_ratio', type=float, default=0.2, help='Selection ratio per round')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for vector computation')
    parser.add_argument('--model_path', type=str, default='./codegen-2B', help='Model path')
    parser.add_argument('--checkpoint_path', type=str, default='checkpointSearch/', help='Checkpoint path')
    parser.add_argument('--diversity_method', type=str, default='min_distance', 
                        choices=['min_distance', 'avg_distance', 'max_similarity'],
                        help='Diversity computation method')
    
    args = parser.parse_args()
    
    # 验证输入参数
    if not args.input_json and not args.seed_dir:
        parser.error("Either --input_json or --seed_dir must be provided")
    if not args.output_json and not args.output_dir:
        parser.error("Either --output_json or --output_dir must be provided")

    
    # 创建采样器（所有进程都需要创建以加载模型）
    sampler = DiversitySampler(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size
    )
    
    # 主进程加载数据
    if sampler.is_main_process:
        # 从 JSON 文件或目录加载数据
        if args.input_json:
            print(f"[Info] Loading programs from JSON file: {args.input_json}")
            with open(args.input_json, 'r', encoding='utf-8') as f:
                all_programs = json.load(f)
            print(f"[Info] Loaded {len(all_programs)} programs from JSON")
        else:
            if args.output_dir and os.path.exists(args.output_dir):
                shutil.rmtree(args.output_dir)
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
            print(f"[Info] Loading programs from directory: {args.seed_dir}")
            all_programs = load_programs_from_directory(args.seed_dir)
            print(f"[Info] Loaded {len(all_programs)} programs from directory")
        
        # 去重
        unique_programs = list(set(all_programs))
        print(f"[Info] Unique programs: {len(unique_programs)}")
    else:
        unique_programs = None
    
    # 同步所有进程，广播数据
    sampler.accelerator.wait_for_everyone()
    
    # 使用 accelerator 广播数据到所有进程
    if sampler.num_processes > 1:
        import torch.distributed as dist
        # 主进程广播数据长度
        if sampler.is_main_process:
            data_len = torch.tensor([len(unique_programs)], device=sampler.device)
        else:
            data_len = torch.tensor([0], device=sampler.device)
        dist.broadcast(data_len, src=0)
        
        # 主进程序列化数据并广播
        if sampler.is_main_process:
            import pickle
            data_bytes = pickle.dumps(unique_programs)
            data_tensor = torch.ByteTensor(list(data_bytes)).to(sampler.device)
            size_tensor = torch.tensor([len(data_bytes)], device=sampler.device)
        else:
            size_tensor = torch.tensor([0], device=sampler.device)
        
        dist.broadcast(size_tensor, src=0)
        
        if not sampler.is_main_process:
            data_tensor = torch.zeros(size_tensor.item(), dtype=torch.uint8, device=sampler.device)
        
        dist.broadcast(data_tensor, src=0)
        
        if not sampler.is_main_process:
            import pickle
            data_bytes = bytes(data_tensor.cpu().tolist())
            unique_programs = pickle.loads(data_bytes)
    
    # 所有进程都执行采样（compute_vectors_batch 会自动分发数据到多卡）
    selected_programs = sampler.sample(
        all_texts=unique_programs,
        target_count=args.target_count,
        initial_count=args.initial_count,
        candidate_count=args.candidate_count,
        select_ratio=args.select_ratio,
        diversity_method=args.diversity_method,
        save_file=args.output_json if sampler.is_main_process else None,  # 只有主进程保存
        save_path=args.output_dir if sampler.is_main_process else None  # 只有主进程保存
    )
    
    if sampler.is_main_process:
        print(f"[Info] Sampling completed!")
        print(f"[Info] Total selected: {len(selected_programs)}")


if __name__ == '__main__':
    main()
