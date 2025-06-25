## Kaggle比赛——Predicting Optimal Fertilizers

### 思路

把所有的参数组织成一个prompt，然后用一个分类大模型训练

```python
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
```

### 结果

训练集75万条数据，数据太大了，8b的参数不行，得是0.3b之类的

0.3b快是快，但效果太差

用8b，8个小时1轮，训练集得分0.15

暂时没有推理，推理的话肯定也要几个小时，太慢了