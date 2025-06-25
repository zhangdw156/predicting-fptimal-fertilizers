from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, get_scheduler
from peft import get_peft_model, LoraConfig, PeftModel
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
import json
import glob
import shutil
import json
import argparse

def generate_fertilizer_prompt(
    temperature: float,
    humidity: float,
    moisture: float,
    soil_type: str,
    crop_type: str,
    nitrogen: float,
    potassium: float,
    phosphorous: float
) -> str:
    """
    生成用于预测肥料类型的prompt
    
    参数:
        temperature: 温度 (°C)
        humidity: 湿度 (%)
        moisture: 土壤湿度 (%)
        soil_type: 土壤类型 (如: Sandy, Loamy, Clayey)
        crop_type: 作物类型 (如: Wheat, Rice, Sugarcane)
        nitrogen: 土壤氮含量 (mg/kg)
        potassium: 土壤钾含量 (mg/kg)
        phosphorous: 土壤磷含量 (mg/kg)
    
    返回:
        格式化的prompt字符串
    """
    return f"""Predict the optimal fertilizer type for the given agricultural parameters:

{{
"Temperature": {temperature},  # Crop environment temperature in Celsius
"Humidity": {humidity},  # Relative humidity percentage
"Soil_Moisture": {moisture},  # Soil moisture content percentage
"Soil_Type": "{soil_type}",  # Soil texture (e.g., Sandy, Loamy, Clayey)
"Crop_Type": "{crop_type}",  # Type of crop (e.g., Wheat, Rice, Sugarcane)
"Nitrogen": {nitrogen},  # Soil nitrogen content in mg/kg
"Potassium": {potassium},  # Soil potassium content in mg/kg
"Phosphorus": {phosphorous}  # Soil phosphorus content in mg/kg
}}
"""
    
# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if accelerator.is_main_process:
        print(f"设置随机种子: {seed}")

# 加载模型与分词器
def load_model_and_tokenizer(model_path, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    config.pad_token_id = tokenizer.pad_token_id
    config.num_labels = num_labels
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    
    return model, tokenizer, config

# 加载并预处理数据
def load_and_preprocess_data(dataset_path, tokenizer, batch_size, max_length):
    # 加载数据
    dataset = load_dataset("csv", data_files=dataset_path)
    dataset = dataset["train"].train_test_split(test_size=0.01, seed=42)
    
    # 数据预处理函数
    def tokenize_function(examples):
        examples["text"] = [
            generate_fertilizer_prompt(
                temp, hum, mois, soil, crop, nit, pot, phos
            )
            for temp, hum, mois, soil, crop, nit, pot, phos in zip(
                examples["Temparature"], examples["Humidity"], examples["Moisture"],
                examples["Soil Type"], examples["Crop Type"], examples["Nitrogen"],
                examples["Potassium"], examples["Phosphorous"]
            )
        ]
        import json
        with open('mapping.json','r') as f:
            mapping=json.load(f)
            name_to_id=mapping['name_to_id']
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        tokenized['labels']=[name_to_id[cat] for cat in examples['Fertilizer Name']]
        
        return tokenized
    
    # 应用预处理
    dataset = dataset.map(tokenize_function, batched=True)
    
    # 自定义collate函数
    def custom_collate(batch):
        batch_dict = {key: [example[key] for example in batch] for key in ["input_ids", "attention_mask", "labels"]}
        for key in batch_dict:
            if isinstance(batch_dict[key][0], list):
                batch_dict[key] = torch.tensor(batch_dict[key])
            elif isinstance(batch_dict[key][0], (int, float)):
                batch_dict[key] = torch.tensor(batch_dict[key])
        return batch_dict
    
    # 创建DataLoader
    train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, collate_fn=custom_collate)
    eval_dataloader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=custom_collate)
    
    return train_dataloader, eval_dataloader, dataset

# 准备训练组件
def prepare_training_components(model, train_dataloader, num_epochs, lr=5e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.05,
        num_training_steps=num_training_steps
    )
    return optimizer, lr_scheduler, num_training_steps

# 加载训练指标
def load_metrics(output_dir):
    if os.path.exists(os.path.join(output_dir,"metrics.json")):
        with open(os.path.join(output_dir, "metrics.json"), "r", encoding="utf-8") as f:
            saved_metrics = json.load(f)
            train_metrics = saved_metrics["train"]
            val_metrics = saved_metrics["val"]
    else:
        train_metrics = {"loss": [], "accuracy": []}
        val_metrics = {"loss": [], "accuracy": []}
    return train_metrics,val_metrics

def save_metrics(train_metrics,val_metrics,output_dir):
    metrics = {"train": train_metrics, "val": val_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

# 加载检查点（如果存在）
def load_checkpoint(model, checkpoint_path,train_dataloader,num_epochs):
    # 仅加载LoRA权重
    model = PeftModel.from_pretrained(model, checkpoint_path)

    optimizer, lr_scheduler, _ = prepare_training_components(model, train_dataloader, num_epochs)

    optimizer_state = torch.load(os.path.join(checkpoint_path, "optimizer.pt"))
    scheduler_state = torch.load(os.path.join(checkpoint_path, "scheduler.pt"))

    optimizer.load_state_dict(optimizer_state)
    lr_scheduler.load_state_dict(scheduler_state)
    print(f"成功加载检查点:{checkpoint_path}")
    return model,optimizer,lr_scheduler
     
# 同步所有进程的全局步数，确保所有进程达到或超过目标步数
def sync_global_step(accelerator, current_step, target_step):
    # 创建一个整数张量，表示当前进程是否达到目标步数
    # 1 表示达到，0 表示未达到
    has_reached_target = torch.tensor(1 if current_step >= target_step else 0, 
                                     dtype=torch.int32, device=accelerator.device)
    
    # 使用all_reduce操作同步所有进程的状态
    # 当所有进程的has_reached_target都为1时，all_reached才为True
    all_reached = torch.tensor(0, dtype=torch.int32, device=accelerator.device)
    
    while all_reached.item() == 0:
        # 使用SUM操作聚合所有进程的状态
        torch.distributed.all_reduce(has_reached_target, op=torch.distributed.ReduceOp.SUM)
        
        # 获取进程总数
        world_size = torch.distributed.get_world_size()
        
        # 如果聚合结果等于进程总数，表示所有进程都达到了目标步数
        all_reached = torch.tensor(1 if has_reached_target.item() == world_size else 0, 
                                  dtype=torch.int32, device=accelerator.device)
        
        # 广播all_reached状态给所有进程
        torch.distributed.broadcast(all_reached, src=0)
        
        # 如果当前进程未达到目标步数，则继续训练
        if has_reached_target.item() < world_size:
            return False
    
    return True

# 保存检查点并绘制指标图（仅主进程执行）
def save_checkpoint(global_step, model, optimizer, lr_scheduler, 
                    checkpoint_dir, steps_per_epoch, accelerator):
    if not accelerator.is_main_process:
        return
    
    current_epoch = global_step // steps_per_epoch
    current_batch = global_step % steps_per_epoch
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-step-{global_step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # 仅保存LoRA权重
    model_to_save = accelerator.unwrap_model(model)
    if hasattr(model_to_save, "module"):
        model_to_save = model_to_save.module
    model_to_save.save_pretrained(checkpoint_path)
    
    # 保存优化器和调度器状态
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))
    
    print(f"已保存检查点到 {checkpoint_path} (Step {global_step+1}, Epoch {current_epoch+1}, Batch {current_batch+1})")
    
    # 管理检查点数量，最多保留2个
    checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-step-*")), 
                             key=lambda x: int(x.split("-")[-1]))
    while len(checkpoint_paths) > 2:
        oldest_checkpoint = checkpoint_paths[0]
        shutil.rmtree(oldest_checkpoint)
        print(f"删除旧检查点: {oldest_checkpoint}")
        checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-step-*")), 
                                 key=lambda x: int(x.split("-")[-1]))

# 绘制训练指标图
def plot_metrics(train_metrics, val_metrics, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_metrics["loss"], label='Train Loss')
    ax1.plot(val_metrics["loss"], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training & Validation Loss')
    
    # 准确率曲线
    ax2.plot(train_metrics["accuracy"], label='Train Accuracy')
    ax2.plot(val_metrics["accuracy"], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.set_title('Training & Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 训练循环
def train_loop(num_epochs, model, train_dataloader, eval_dataloader, 
               optimizer, lr_scheduler, train_metrics, val_metrics, global_step, 
               checkpoint_interval, checkpoint_dir, output_dir, accelerator):
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch
    start_epoch = global_step // steps_per_epoch
    start_batch_in_epoch = global_step % steps_per_epoch
    
    for epoch in range(start_epoch, num_epochs):
        # 训练阶段
        model.train()
        
        train_total_loss = 0
        train_correct = 0
        train_total_samples = 0
        
        # 创建原始迭代器
        data_iter = enumerate(train_dataloader)

        # 用tqdm包装迭代器
        if accelerator.is_main_process:
            progress_bar = tqdm(data_iter, total=steps_per_epoch,desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            progress_bar=data_iter
        
        # 如果是断点续训的起始epoch，跳过已训练的batches
        if epoch == start_epoch:
            for i in range(start_batch_in_epoch):
                next(data_iter)
                if accelerator.is_main_process:
                    progress_bar.update(1)  # 手动更新进度条
            print("断点训练跳过之前的batch")
        
        for batch_idx, batch in progress_bar:
            current_step = global_step + 1
            
            # 前向传播
            with accelerator.autocast():
                outputs = model(**batch)
            loss = outputs.loss
                
            # 反向传播
            accelerator.backward(loss)
            
            # 优化器步骤
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 计算训练集准确率
            logits = outputs.logits
            predictions = logits.argmax(dim=-1)
            labels = batch["labels"]

            # 收集所有进程的结果
            predictions = accelerator.gather(predictions)
            labels = accelerator.gather(labels)
            
            train_correct += (predictions == labels).sum().item()
            train_total_samples += labels.size(0)
            train_total_loss += loss.item() * labels.size(0)
            
            # 更新全局步数
            global_step += 1
            
            # 检查是否需要保存检查点
            if global_step % checkpoint_interval == 0:
                # 同步所有进程，确保所有进程都达到或超过当前全局步数
                all_reached = sync_global_step(accelerator, global_step, global_step)
                
                if all_reached:
                    # 所有进程都达到了目标步数，可以保存检查点
                    accelerator.wait_for_everyone()
                    save_checkpoint(global_step, model, optimizer, lr_scheduler, 
                                                 checkpoint_dir, steps_per_epoch, accelerator)

        # 计算 epoch 平均损失和准确率
        epoch_loss = train_total_loss / train_total_samples if train_total_samples > 0 else 0
        epoch_accuracy = train_correct / train_total_samples if train_total_samples > 0 else 0
        
        train_metrics["loss"].append(epoch_loss)
        train_metrics["accuracy"].append(epoch_accuracy)

        # 验证阶段
        model.eval()
        val_total_loss = 0
        val_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                with accelerator.autocast():
                    outputs = model(**batch)
                val_loss = outputs.loss
                logits = outputs.logits
                
                # 计算准确率
                predictions = logits.argmax(dim=-1)
                labels = batch["labels"]
                
                # 收集所有进程的结果
                predictions = accelerator.gather(predictions)
                labels = accelerator.gather(labels)
                
                val_correct += (predictions == labels).sum().item()
                val_total_samples += labels.size(0)
                val_total_loss += val_loss.item() * batch["input_ids"].size(0)
        
        # 计算验证集指标
        val_epoch_loss = val_total_loss / val_total_samples if val_total_samples > 0 else 0
        val_accuracy = val_correct / val_total_samples if val_total_samples > 0 else 0
        
        val_metrics["loss"].append(val_epoch_loss)
        val_metrics["accuracy"].append(val_accuracy)
        
        # 打印训练和验证进度（仅主进程）
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {epoch_loss:.4f} - "
                  f"Train Accuracy: {epoch_accuracy:.4f} - "
                  f"Val Loss: {val_epoch_loss:.4f} - "
                  f"Val Accuracy: {val_accuracy:.4f}")
        
        # 每个epoch结束后保存检查点
        # 同步所有进程，确保所有进程都完成了当前epoch
        all_reached = sync_global_step(accelerator, global_step, global_step)
        
        if all_reached:
            accelerator.wait_for_everyone()
            save_metrics(train_metrics,val_metrics,output_dir)
            save_checkpoint(global_step, model, optimizer, lr_scheduler, 
                            checkpoint_dir, steps_per_epoch, accelerator)
    
    return train_metrics, val_metrics, global_step

# 保存最终模型和指标（仅主进程执行）
def save_final_model(model, tokenizer, output_dir, accelerator):
    
    print("保存最终模型...")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    final_model = unwrapped_model.module if hasattr(unwrapped_model, "module") else unwrapped_model
    os.makedirs(output_dir, exist_ok=True)
    
    # 只保存LoRA权重
    final_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def parse_args():
    parser = argparse.ArgumentParser(description="训练模型的命令行参数")
    
    # 添加命令行参数
    parser.add_argument('--model_path', type=str, default="/data/download-model/DeepSeek-R1-0528-Qwen3-8B",
                        help='模型路径')
    parser.add_argument('--dataset_path', type=str, default="data/train.csv",
                        help='数据集路径')
    parser.add_argument('--output_dir', type=str, default="finetuned_model",
                        help='输出路径')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练的轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='保存检查点的间隔步数')
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                        help='检查点目录')
    parser.add_argument('--num_labels', type=int, default="2",
                        help='类别数量')
    parser.add_argument('--max_length', type=int, default="2048",
                        help='最大token长度')
    
    # 解析参数
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()

    model_path = args.model_path
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    checkpoint_interval = args.checkpoint_interval
    checkpoint_dir = args.checkpoint_dir
    num_labels = args.num_labels
    max_length=args.max_length
    
    os.makedirs(output_dir,exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化Accelerator
    # accelerator = Accelerator(mixed_precision="bf16")
    accelerator = Accelerator()
    device = accelerator.device
    if accelerator.is_main_process:
        print(f"使用设备: {device}")
        print(f"模型路径: {model_path}")
        print(f"数据集路径: {dataset_path}")
        print(f"输出路径: {output_dir}")
        print(f"训练轮数: {num_epochs}")
        print(f"批次大小: {batch_size}")
        print(f"检查点间隔: {checkpoint_interval}")
        print(f"检查点路径: {checkpoint_dir}")
        print(f"类别数量: {num_labels}")
        print(f"最大token长度: {max_length}")

    # 初始化训练指标
    train_metrics = {"loss": [], "accuracy": []}
    val_metrics = {"loss": [], "accuracy": []}
    global_step = 0
    start_epoch = 0
    start_batch_in_epoch = 0
    
    # 设置随机种子
    set_seed(42)

    # 加载模型与分词器
    model, tokenizer, _ = load_model_and_tokenizer(model_path, num_labels)

    # 加载并预处理数据
    train_dataloader, eval_dataloader, dataset = load_and_preprocess_data(dataset_path, tokenizer, batch_size,max_length)
    
    # 加载检查点（如果存在）
    if os.path.exists(checkpoint_dir) and len(glob.glob(os.path.join(checkpoint_dir, "checkpoint-step-*"))) > 0:
        
        checkpoint_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-step-*")), 
                                key=lambda x: int(x.split("-")[-1]))
        latest_checkpoint = checkpoint_paths[-1]

        train_metrics,val_metrics=load_metrics(output_dir)

        global_step = int(latest_checkpoint.split("-")[-1])

        model,optimizer,lr_scheduler = load_checkpoint(
            model, latest_checkpoint,train_dataloader,num_epochs
        )
    else:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="SEQ_CLS",
            target_modules=["q_proj", "k_proj"]
        )
        model = get_peft_model(model, lora_config)
        optimizer, lr_scheduler, _ = prepare_training_components(model, train_dataloader, num_epochs)

    # 使用Accelerator准备训练组件
    train_dataloader, eval_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer, lr_scheduler
    )
    
    # 执行训练
    if accelerator.is_main_process:
        print("开始训练...")
    train_metrics, val_metrics, global_step = train_loop(
        num_epochs, model, train_dataloader, eval_dataloader,
        optimizer, lr_scheduler, train_metrics, val_metrics, global_step,
        checkpoint_interval, checkpoint_dir, output_dir, accelerator
    )
    
    # 保存最终模型和指标
    save_final_model(model, tokenizer, output_dir, accelerator)
    # 绘制最终指标图
    plot_metrics(train_metrics, val_metrics, os.path.join(output_dir, "final_training_metrics.png"))
    print("可视化训练指标完成")