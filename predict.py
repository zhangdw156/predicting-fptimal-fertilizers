import json
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline,AutoConfig
from peft import PeftModel
import torch
import csv
import argparse
import pandas as pd

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
    
def parse_args():
    parser = argparse.ArgumentParser(description="训练模型的命令行参数")
    
    # 添加命令行参数
    parser.add_argument('--model_path', type=str, default="/data/download-model/DeepSeek-R1-0528-Qwen3-8B",
                        help='模型路径')
    parser.add_argument('--input_file', type=str, default="data/test.csv",
                        help='测试集')
    parser.add_argument('--output_file', type=str, default="submission.csv",
                        help='输出文件')
    parser.add_argument('--lora_path', type=str, default="finetuned_model",
                        help='lora路径')
    parser.add_argument('--num_labels', type=int, default="2",
                        help='类别数量')
    parser.add_argument('--max_length', type=int, default="2048",
                        help='最大token长度')
    
    # 解析参数
    args = parser.parse_args()
    return args

args=parse_args()

model_path=args.model_path
lora_path=args.lora_path
input_file=args.input_file
output_file=args.output_file
num_labels=args.num_labels
max_length=args.max_length

# 加载模型与分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
config.pad_token_id = tokenizer.pad_token_id  # 关键步骤
config.num_labels = num_labels
model = AutoModelForSequenceClassification.from_pretrained(model_path,config=config)
model = PeftModel.from_pretrained(model, lora_path)

model = model.merge_and_unload()

classifer = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=max_length,
    top_k=3
)

sentences=[]
ids=[]
with open(input_file,"r",encoding="utf-8") as f:
    reader=csv.reader(f)
    next(reader)
    for row in reader:
        sentences.append(generate_fertilizer_prompt(*row[1:]))
        ids.append(row[0])

id_to_name=[]
with open('mapping.json','r') as f:
    mapping=json.load(f)
    id_to_name=mapping['id_to_name']

outputs=classifer(sentences)

df = pd.DataFrame(outputs)
df['Fertilizer Name']=df['label'].str.split('_').str[-1].apply(lambda x:id_to_name(int(x)))
df['id']=ids
# # 写入CSV文件
df.to_csv(output_file, sep=',', columns=['id','Fertilizer Name'],header=True,index=False)





