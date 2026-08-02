# CareerMind 分析包（big_bag）

这个目录用于 GitHub 展示，按“代码 / 数据”分离：

- `code/`：核心分析脚本（数据处理、聚类、回归）。
- `data/`：输入数据与分析产出数据。

已确保 `code/` 中脚本默认从 `data/` 读取，并将输出写回 `data/`。

请注意配置您自己的Deepseek API Key 在`code/job_clustering.py`中

## 目录说明

### 1) 代码目录 `code/`

- `temp.py`  
  从 `data/jobs().csv` 读取原始职位数据，提取 `other_requirement` 字段，输出 `data/jobs(1).csv`。

- `etl.py`  
  从 `data/jobs(1).csv` 读取，完成经验数字化、技能合并、薪资标准化、向量化，输出：
  - `data/job_vec.csv`
  - `data/skill_merge_preview.csv`

- `job_clustering.py`  
  从 `data/job_vec.csv` 读取，按行业聚类画像，输出到：
  - `data/clustered_output/cluster_profiles.csv`
  - `data/clustered_output/clustered_*.csv`
  - 以及聚类提示词与缓存文件（同目录下的 `*.txt`、`llm_cache.json`）

- `salary_regression.py`  
  从 `data/job_vec.csv` 读取，按行业做薪资分析，输出到：
  - `data/regression_output/exp_curve.csv`
  - `data/regression_output/skill_impact.csv`
  - `data/regression_output/skill_value_robust.csv`

### 2) 数据目录 `data/`

#### 核心输入数据

- `jobs().csv`：原始数据表。  
- `jobs(1).csv`：`temp.py` 处理后数据（新增 `other_requirement`）。  
- `job_vec.csv`：`etl.py` 处理后向量化主表。  
- `skill_merge_preview.csv`：`etl.py` 生成的高相关技能合并预览表。  

#### 聚类输出 `data/clustered_output/`

- `cluster_profiles.csv`：各行业聚类画像摘要。  
- `clustered_*.csv`：分行业聚类结果（含组内聚类 ID）。  

#### 回归输出 `data/regression_output/`

- `exp_curve.csv`：大样本行业（>=70）经验-薪资曲线。  
- `skill_impact.csv`：大样本行业技能系数（回归结果）。  
- `skill_value_robust.csv`：中样本行业（10~69）稳健技能价值估计。  

## 推荐执行顺序（不要求现在运行）

在 `big_bag/code` 下按顺序运行：

1. `python temp.py`
2. `python etl.py`
3. `python job_clustering.py`
4. `python salary_regression.py`

## 说明

- 本目录为展示整理版本，未改动 `big_bag` 外文件。  
- 如果需要完全复现实验，请确保 Python 依赖和本地模型文件可用（如 sentence-transformers / gensim / sklearn / statsmodels）。  
