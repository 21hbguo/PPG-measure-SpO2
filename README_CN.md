# 基于PPG的SpO2预测框架
本项目提供一个可扩展的血氧饱和度（SpO2）预测框架。系统从原始视频中提取光电容积脉搏波（PPG）信号，并基于这些信号完成SpO2估计。仓库包含数据预处理、模型训练、结果推理与交互式可视化应用，便于围绕不同深度学习模型开展实验。

## 主要特性
- 完整流程覆盖：从原始视频处理、信号构建、模型训练到最终SpO2预测
- 模型可扩展：可在配置文件中切换或新增不同模型结构
- 结果可复现：关键路径、超参数与随机种子集中在 `config.py`
- 输出结构清晰：训练权重、评估结果和可视化输出按模型分类保存
- 支持网页应用：提供基于 Streamlit 的交互式分析与预测界面

## 项目流程
### 1. 数据预处理 `preprocess_data.py`
- 输入：原始视频文件与真值CSV
- 处理：提取RGB信号、同步时间戳、清洗并整理训练数据
- 输出：可直接用于训练的预处理CSV

### 2. 模型训练 `train.py`
- 输入：预处理后的CSV数据
- 处理：完成数据加载、标准化、模型训练与性能评估
- 输出：模型权重、标准化器、损失曲线和评估结果

### 3. 推理预测 `predict.py`
- 输入：待预测CSV与已训练模型
- 处理：加载标准化器和模型权重，对输入数据执行推理
- 输出：SpO2预测结果及可视化图表

## 项目结构
```text
PPG-measure-SpO2/
├── data/
│   ├── gt/
│   ├── raw_videos/
│   └── processed/
├── models/
├── results/
├── preprocess_data.py
├── train.py
├── predict.py
├── app.py
├── config.py
├── data_utils.py
├── utils.py
├── requirements.txt
├── README.md
└── README_CN.md
```

## 环境配置
```bash
conda create --name spo2 python=3.13
conda activate spo2
pip install -r requirements.txt
```

## 使用步骤
### 1. 准备数据
- 将原始视频放入 `data/raw_videos/`
- 将对应真值CSV放入 `data/gt/`

### 2. 执行预处理
```bash
python preprocess_data.py
```

### 3. 配置并训练模型
- 在 `config.py` 中设置数据路径、模型名称和训练参数
- 运行：
```bash
python train.py
```

### 4. 执行预测
- 在 `predict.py` 中指定待预测CSV
- 确认 `config.py` 中模型名称与训练权重一致
- 运行：
```bash
python predict.py
```

## Web应用
项目提供 `app.py` 作为交互式网页界面，可用于上传视频、选择训练模型、执行推理并查看可视化结果。

启动方式：
```bash
streamlit run app.py
```

## 扩展模型
如需新增模型，可按以下方式扩展：
1. 在 `models/` 下新增模型定义文件
2. 在模型注册位置加入新模型
3. 在 `config.py` 中补充模型配置
4. 切换 `MODEL_NAME` 后重新训练
