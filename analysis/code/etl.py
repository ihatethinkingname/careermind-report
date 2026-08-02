# from pyspark.sql import SparkSession
# from pyspark.sql.functions import *
import glob
import os
from collections import Counter
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INPUT_CSV = os.path.join(DATA_DIR, "jobs(1).csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "job_vec.csv")
SKILL_MERGE_PREVIEW_CSV = os.path.join(DATA_DIR, "skill_merge_preview.csv")
SKILL_MERGE_CORRELATION_THRESHOLD = 0.75

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
import re
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("TalentGraph ETL") \
#     .enableHiveSupport() \
#     .getOrCreate()

# # MySQL connection properties
# mysql_properties = {
#     "user": "root",
#     "password": "password",
#     "driver": "com.mysql.cj.jdbc.Driver"
# }

# # Extract data from MySQL
# def extract_from_mysql():
#     # Extract jobs data
#     jobs_df = spark.read.jdbc(
#         url="jdbc:mysql://mysql:3306/careermind",
#         table="jobs",
#         properties=mysql_properties
#     )
    
    # # Extract users data (assuming users table exists)
    # users_df = spark.read.jdbc(
    #     url="jdbc:mysql://mysql:3306/careermind",
    #     table="users",
    #     properties=mysql_properties
    # )
    
    # # Extract interactions data (assuming interactions table exists)
    # interactions_df = spark.read.jdbc(
    #     url="jdbc:mysql://mysql:3306/talentgraph",
    #     table="interactions",
    #     properties=mysql_properties
    # )

    # # Extract static report source data for downstream ETL analysis
    # static_source_df = spark.read.jdbc(
    #     url="jdbc:mysql://mysql:3306/careermind",
    #     table="report_static_source",
    #     properties=mysql_properties
    # )
    
    # return jobs_df, users_df, interactions_df, static_source_df
    # return jobs_df

# experience_requirement to numeric
def exp_to_numeric(text):
    # mapping = {
    #     "无需经验": 0,
    #     "1年以内": 0.5,
    #     "1-3年": 2,
    #     "3-5年": 4,
    #     "5年及以上": 5,
    #     "10年以上": 10
    # }
    # # 正则提取也可，此处简化
    # return mapping.get(text, 0) / 10.0  # 归一化到0-1
    text = text.strip()
    # 1. 无需经验
    if "无需经验" in text:
        return 0.0 / 10.0
    # 2. 1年以内 -> 0.5
    if "1年以内" in text or "1年内" in text:
        return 0.5 / 10.0
    # 3. 范围型：n-m年
    match = re.search(r'(\d+)\s*-\s*(\d+)\s*年', text)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        avg = (low + high) / 2.0
        return avg / 10.0
    
    # 4. n年及以上
    match = re.search(r'(\d+)\s*年\s*及以上', text)
    if match:
        n = int(match.group(1))
        return n / 10.0   # 取 n
    
    # 5. n年以上
    match = re.search(r'(\d+)\s*年\s*以上', text)
    if match:
        n = int(match.group(1))
        return (n + 1) / 10.0   # 取 n+1
    
    # 6. 单独数字年（如 "3年"）
    match = re.search(r'(\d+)\s*年', text)
    if match:
        n = int(match.group(1))
        return n / 10.0
    
    # 默认未识别 -> 0
    return 0.0


def salary_to_yuan_per_month(val, unit, period):
    if pd.isna(val):
        return np.nan
    unit = str(unit).strip() if pd.notna(unit) else '元'
    period = str(period).strip() if pd.notna(period) else '月'

    unit_multipliers = {
        '万': 10000.0,
        '千': 1000.0,
        '元': 1.0
    }
    period_multipliers = {
        '月': 1.0,
        '年': 1.0 / 12.0,
        '天': 30.0,
        '单': 1.0
    }

    # Unknown labels fallback to base assumptions: 元 / 月
    val = float(val) * unit_multipliers.get(unit, 1.0)
    val = val * period_multipliers.get(period, 1.0)
    return val


def extract_skills(skill_str):
    if pd.isna(skill_str):
        return []
    return [token.strip() for token in re.split(r'[,;；]+', str(skill_str)) if token.strip()]


def build_skill_matrix(df_group):
    all_skills = []
    for skills in df_group['skills_list']:
        all_skills.extend(skills)

    skill_counts = Counter(all_skills)
    valid_skills = [skill for skill, count in skill_counts.items() if count >= 2]
    if not valid_skills:
        return pd.DataFrame(index=df_group.index), {}, skill_counts

    matrix = pd.DataFrame(0, index=df_group.index, columns=valid_skills)
    for idx, skills in enumerate(df_group['skills_list']):
        for skill in skills:
            if skill in matrix.columns:
                matrix.loc[df_group.index[idx], skill] = 1
    return matrix, {skill: skill for skill in valid_skills}, skill_counts


def compute_skill_merge_preview(df, threshold=SKILL_MERGE_CORRELATION_THRESHOLD):
    preview_rows = []
    for industry, df_group in df.groupby('industry_group'):
        skill_matrix, _, skill_counts = build_skill_matrix(df_group)
        if skill_matrix.shape[1] < 2:
            continue

        corr = skill_matrix.corr(method='pearson')
        cols = list(skill_matrix.columns)
        edges = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r) and r > threshold:
                    edges.append((cols[i], cols[j]))
        if not edges:
            continue

        parent = {}

        def find(x):
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in edges:
            union(a, b)

        groups = {}
        for col in cols:
            groups.setdefault(find(col), []).append(col)

        for members in groups.values():
            if len(members) < 2:
                continue
            rep_skill = max(members, key=lambda s: (skill_counts[s], s))
            preview_rows.append({
                'industry_group': industry,
                'suggested_label': f"{rep_skill}技能包",
                'original_skills': ';'.join(members)
            })
    return preview_rows


def apply_skill_merges(df, preview_rows):
    if not preview_rows:
        return df
    merge_map = {}
    for row in preview_rows:
        industry = row['industry_group']
        merge_map.setdefault(industry, {})
        label = row['suggested_label']
        for skill in row['original_skills'].split(';'):
            merge_map[industry][skill] = label

    def merge_skills(row):
        industry = row['industry_group']
        original = extract_skills(row.get('job_skills', ''))
        merged = [merge_map.get(industry, {}).get(skill, skill) for skill in original]
        seen = set()
        unique = []
        for skill in merged:
            if skill not in seen:
                seen.add(skill)
                unique.append(skill)
        return ','.join(unique)

    result = df.copy()
    result['merged_job_skills'] = result.apply(merge_skills, axis=1)
    return result

WORD_VECTOR_MODEL = None
WORD_VECTOR_MODEL_LOADED = False
SENTENCE_MODEL = None
LOCAL_MODEL_ROOT = os.path.join(BASE_DIR, '..', 'local_model')
WORD_VECTOR_MODEL_PATH = os.environ.get(
    'WORD_VECTOR_MODEL_PATH',
    os.path.join(LOCAL_MODEL_ROOT, 'sgns.zhihu.word')
)
LOCAL_MODEL_DIR = os.path.join(LOCAL_MODEL_ROOT, 'paraphrase-multilingual-MiniLM-L12-v2')
SENTENCE_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# job_skills to vectors
def find_word_vector_file(path):
    if os.path.isdir(path):
        for entry in os.listdir(path):
            candidate = os.path.join(path, entry)
            if os.path.isfile(candidate) and (entry.endswith('.word') or entry.endswith('.txt') or entry.endswith('.bin')):
                return candidate
        return None
    if os.path.isfile(path):
        return path
    return None


def get_word_vector_model(path=WORD_VECTOR_MODEL_PATH):
    global WORD_VECTOR_MODEL, WORD_VECTOR_MODEL_LOADED
    if WORD_VECTOR_MODEL_LOADED:
        return WORD_VECTOR_MODEL

    resolved_path = find_word_vector_file(path)
    if resolved_path is None:
        candidates = [
            os.path.join(LOCAL_MODEL_ROOT, 'sgns.zhihu.word'),
            os.path.join(LOCAL_MODEL_ROOT, 'sgns.zhihu.word.txt'),
            os.path.join(LOCAL_MODEL_ROOT, 'sgns.zhihu.word.bz2'),
            os.path.join(LOCAL_MODEL_ROOT, 'tencent-ailab-embedding-zh-d100.bin')
        ]
        for candidate in candidates:
            resolved_path = find_word_vector_file(candidate)
            if resolved_path is not None:
                print(f"Using local word vector model candidate: {resolved_path}")
                break

    if resolved_path is None:
        print(f"Word vector model not found at {path}, skipping word vectors")
        WORD_VECTOR_MODEL_LOADED = True
        return None

    try:
        print(f"Loading word vector model from {resolved_path}")
        binary_mode = resolved_path.endswith('.bin')
        try:
            WORD_VECTOR_MODEL = KeyedVectors.load_word2vec_format(resolved_path, binary=binary_mode)
        except Exception as e_binary:
            print(f"Binary load failed, retrying as text format: {e_binary}")
            WORD_VECTOR_MODEL = KeyedVectors.load_word2vec_format(resolved_path, binary=False)
    except Exception as e:
        print(f"Failed to load word vector model: {e}")
        WORD_VECTOR_MODEL = None
    WORD_VECTOR_MODEL_LOADED = True
    return WORD_VECTOR_MODEL


def has_model_weights(model_dir):
    if not os.path.isdir(model_dir):
        return False
    for weight_name in [
        'pytorch_model.bin',
        'model.safetensors',
        'pytorch_model.safetensors',
        'tf_model.h5'
    ]:
        if os.path.exists(os.path.join(model_dir, weight_name)):
            return True
    return False


def clear_hf_cache_locks(model_name=SENTENCE_MODEL_NAME):
    lock_dir = os.path.expanduser(
        f'~/.cache/huggingface/hub/.locks/models--sentence-transformers--{model_name}'
    )
    if os.path.isdir(lock_dir):
        for lock_file in glob.glob(os.path.join(lock_dir, '*')):
            try:
                os.remove(lock_file)
            except Exception:
                pass
        print(f"Cleared stale HF lock files under {lock_dir}")


def find_local_sentence_model(model_name=SENTENCE_MODEL_NAME):
    if has_model_weights(LOCAL_MODEL_DIR):
        return LOCAL_MODEL_DIR

    repo_root = os.path.expanduser(
        f'~/.cache/huggingface/hub/models--sentence-transformers--{model_name}'
    )
    if os.path.isdir(repo_root):
        snapshots_dir = os.path.join(repo_root, 'snapshots')
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(
                [
                    os.path.join(snapshots_dir, name)
                    for name in os.listdir(snapshots_dir)
                    if os.path.isdir(os.path.join(snapshots_dir, name))
                ]
            )
            for snapshot in snapshots[::-1]:
                if has_model_weights(snapshot):
                    return snapshot

        candidate = os.path.join(repo_root, '0.0.0')
        if has_model_weights(candidate):
            return candidate

        for root, dirs, files in os.walk(repo_root):
            if has_model_weights(root):
                return root

    return None


def get_sentence_model(model_name=SENTENCE_MODEL_NAME):
    global SENTENCE_MODEL
    if SENTENCE_MODEL is None:
        clear_hf_cache_locks(model_name)
        local_model = find_local_sentence_model(model_name)
        if local_model is not None:
            print(f"Loading local sentence model from {local_model}")
            SENTENCE_MODEL = SentenceTransformer(local_model)
        else:
            print(f"Downloading sentence model {model_name} to {LOCAL_MODEL_DIR}")
            try:
                snapshot_download(model_name, local_dir=LOCAL_MODEL_DIR)
                SENTENCE_MODEL = SentenceTransformer(LOCAL_MODEL_DIR)
            except Exception as e:
                print(f"Failed to download or load sentence model: {e}")
                raise
    return SENTENCE_MODEL


def normalize_tags(tag_list):
    if isinstance(tag_list, str):
        return [t.strip() for t in re.split(r'[,;；\s]+', tag_list) if t.strip()]
    if isinstance(tag_list, (list, tuple, np.ndarray, pd.Series)):
        return [str(t).strip() for t in tag_list if str(t).strip()]
    return []


def get_tag_embedding(tag_list):
    tags = normalize_tags(tag_list)
    wv = get_word_vector_model()
    if wv is not None:
        vectors = [wv[tag] for tag in tags if tag in wv]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        # If word vector model exists but no tags are found, stay in the same embedding space.
        return np.zeros(wv.vector_size)

    model = get_sentence_model()
    if len(tags) == 0:
        return np.zeros(model.get_embedding_dimension())

    tag_sentence = ' '.join(tags)
    vec = model.encode([tag_sentence], batch_size=16, show_progress_bar=False)
    return vec[0]


# other_requirement to vector
def get_other_embedding(other_requirement):
    model = get_sentence_model()
    if isinstance(other_requirement, str):
        sentences = [other_requirement]
    elif isinstance(other_requirement, (list, tuple, np.ndarray, pd.Series)):
        sentences = [str(x) for x in other_requirement if str(x).strip()]
    else:
        sentences = []

    if len(sentences) == 0:
        return np.zeros(model.get_embedding_dimension())

    other_vec = model.encode(sentences, batch_size=16, show_progress_bar=False)
    job_other_vec = np.mean(other_vec, axis=0)
    return job_other_vec


def normalize_other_requirement(other_requirement):
    if isinstance(other_requirement, str):
        text = other_requirement.strip()
        return [text] if text else []
    if isinstance(other_requirement, (list, tuple, np.ndarray, pd.Series)):
        return [str(x).strip() for x in other_requirement if str(x).strip()]
    return []


def get_other_embeddings(other_requirements, batch_size=16):
    model = get_sentence_model()
    rows = list(other_requirements)
    groups = [normalize_other_requirement(row) for row in rows]
    all_sentences = [sent for group in groups for sent in group]
    if len(all_sentences) == 0:
        return np.zeros((len(rows), model.get_embedding_dimension()))

    all_vecs = model.encode(all_sentences, batch_size=batch_size, show_progress_bar=False)
    row_vectors = []
    index = 0
    for group in groups:
        if len(group) == 0:
            row_vectors.append(np.zeros(model.get_embedding_dimension()))
        else:
            row_vectors.append(np.mean(all_vecs[index:index + len(group)], axis=0))
            index += len(group)

    return np.vstack(row_vectors)


# Combine three vector together
def job_to_vector_single(exp_text, tags, other_requirement):
    exp_val = exp_to_numeric(str(exp_text))
    skill_vec = get_tag_embedding(tags)
    other_vec = get_other_embedding(other_requirement)
    return np.concatenate([[exp_val], skill_vec, other_vec])


def job_to_vector(exp_text, tags, other_requirement):
    if isinstance(exp_text, (pd.Series, np.ndarray, list)):
        if not isinstance(exp_text, pd.Series):
            exp_text = pd.Series(exp_text)
        tags_series = pd.Series(tags) if not isinstance(tags, pd.Series) else tags
        other_series = pd.Series(other_requirement) if not isinstance(other_requirement, pd.Series) else other_requirement

        exp_vals = exp_text.apply(lambda x: exp_to_numeric(str(x))).to_numpy()
        skill_vectors = np.vstack([get_tag_embedding(tag) for tag in tags_series])
        other_vectors = get_other_embeddings(other_series, batch_size=16)
        return np.hstack([exp_vals.reshape(-1, 1), skill_vectors, other_vectors])

    return job_to_vector_single(exp_text, tags, other_requirement)


def transform_jobs_to_vector_table(df):
    """Return a DataFrame with one vector row per job record."""
    if df.empty:
        return df
    transformed_df = df.copy()
    transformed_df['exp_numeric'] = transformed_df['experience_required'].fillna('').astype(str).apply(exp_to_numeric)
    transformed_df['salary_min_norm'] = transformed_df.apply(
        lambda row: salary_to_yuan_per_month(
            row['salary_min'],
            row.get('salary_unit', '元'),
            row.get('salary_period', '月')
        ),
        axis=1
    )
    transformed_df['salary_max_norm'] = transformed_df.apply(
        lambda row: salary_to_yuan_per_month(
            row['salary_max'],
            row.get('salary_unit', '元'),
            row.get('salary_period', '月')
        ),
        axis=1
    )
    transformed_df['avg_salary'] = (transformed_df['salary_min_norm'] + transformed_df['salary_max_norm']) / 2
    transformed_df['skills_list'] = transformed_df['job_skills'].apply(extract_skills)
    preview_rows = compute_skill_merge_preview(transformed_df)
    pd.DataFrame(preview_rows).to_csv(
        SKILL_MERGE_PREVIEW_CSV,
        index=False,
        encoding="utf-8-sig"
    )
    transformed_df = apply_skill_merges(transformed_df, preview_rows)
    skill_source_col = 'merged_job_skills' if 'merged_job_skills' in transformed_df.columns else 'job_skills'

    vector_matrix = job_to_vector(
        transformed_df['experience_required'],
        transformed_df[skill_source_col],
        transformed_df['other_requirement']
    )
    vector_cols = [f'job_vec_{i}' for i in range(vector_matrix.shape[1])]
    vector_df = pd.DataFrame(vector_matrix, columns=vector_cols, index=transformed_df.index)
    return pd.concat([transformed_df.reset_index(drop=True), vector_df.reset_index(drop=True)], axis=1)

# # 示例
# job_vec = job_to_vector(
#     tags=["团队管理", "npi", "电气自动化"],
#     exp_text="5年及以上",
#     other_sentences=["热情活泼", "抗压能力强"]
# )
# print(job_vec.shape)

# # Transform data
# # def transform_data(jobs_df, users_df, interactions_df, static_source_df):
# def transform_data(jobs_df):
#     # Transform jobs data to job_feature_table (MySQL jobs uses city, min/max salary, required_skills, etc.)
#     job_feature_df = jobs_df.withColumn(
#         "salary_min",
#         col("min_salary").cast("double")
#     ).withColumn(
#         "salary_max",
#         col("max_salary").cast("double")
#     ).withColumn(
#         "skills_required",
#         when(
#             col("required_skills").isNull() | (trim(col("required_skills")) == lit("")),
#             array().cast("array<string>")
#         ).otherwise(split(trim(col("required_skills")), ","))
#     ).withColumn(
#         "experience_required",
#         regexp_extract(col("experience_requirement"), r"(\d+)", 1).cast("int")
#     ).withColumn(
#         "education_required",
#         col("education_requirement")
#     ).withColumn(
#         "industry",
#         when(col("industry").isNull() | (trim(col("industry")) == lit("")), lit("未分类")).otherwise(trim(col("industry")))
#     ).withColumn(
#         "location",
#         col("city")
#     ).select(
#         col("job_id"),
#         col("job_title"),
#         col("company_name"),
#         col("salary_min"),
#         col("salary_max"),
#         col("location"),
#         col("industry"),
#         col("skills_required"),
#         col("experience_required"),
#         col("education_required"),
#         col("job_description"),
#         current_timestamp().alias("created_at"),
#         current_timestamp().alias("updated_at")
#     )
    
    # # Transform users data to user_feature_table
    # user_feature_df = users_df.withColumn(
    #     "skills",
    #     split(col("skills"), ",")
    # ).withColumn(
    #     "industry_preference",
    #     split(col("industry_preference"), ",")
    # ).withColumn(
    #     "location_preference",
    #     split(col("location_preference"), ",")
    # ).select(
    #     col("user_id"),
    #     col("age"),
    #     col("education"),
    #     col("work_experience"),
    #     col("skills"),
    #     col("industry_preference"),
    #     col("salary_expectation"),
    #     col("location_preference"),
    #     col("created_at"),
    #     current_timestamp().alias("updated_at")
    # )
    
    # # Transform interactions data to interaction_table
    # interaction_df = interactions_df.select(
    #     col("id").alias("interaction_id"),
    #     col("user_id"),
    #     col("job_id"),
    #     col("interaction_type"),
    #     col("interaction_time"),
    #     col("created_at")
    # )
    
    # return job_feature_df, user_feature_df, interaction_df, static_source_df
    # return job_feature_df

# Load data to Hive
# def load_to_hive(job_feature_df, user_feature_df, interaction_df, static_source_df):

# def load_to_hive(job_vec):
#     # Load job features
#     job_vec.write.mode("overwrite").saveAsTable("job_vec_table")
    
    # # Load user features
    # user_feature_df.write.mode("overwrite").saveAsTable("user_feature_table")
    
    # # Load interactions
    # interaction_df.write.mode("overwrite").saveAsTable("interaction_table")

    # # Load static report source data for downstream analytics
    # static_source_df.write.mode("overwrite").saveAsTable("report_static_source")

def load_to_csv(job_vec):
    # Load job features
    job_vec.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

# def extract_from_mysql():
#     jobs_df = spark.read.jdbc(
#         url="jdbc:mysql://mysql:3307/careermind",
#         table="jobs",
#         properties=mysql_properties
#     )

#     # 查看 jobs 表结构
#     jobs_df.printSchema()

#     # 查看前几行数据
#     jobs_df.show(20, False)

#     # 查看行数
#     print("jobs row count:", jobs_df.count())

# Main ETL process
def main():
    print("Starting ETL process...")
    
    # Extract
    df=pd.read_csv(INPUT_CSV)
    # jobs_df, users_df, interactions_df, static_source_df = extract_from_mysql()
    # print("Data extracted from MySQL")
    
    # # Transform
    job_vec_df = transform_jobs_to_vector_table(df)
    # job_feature_df, user_feature_df, interaction_df, static_source_df = transform_data(
    #     jobs_df, users_df, interactions_df, static_source_df
    # )
    # print("Data transformed")
    
    # # Load
    load_to_csv(job_vec_df)
    print(f"Transformed {len(job_vec_df)} records into vector table.")
    # load_to_hive(job_feature_df, user_feature_df, interaction_df, static_source_df)
    # print("Data loaded to Hive")
    
    # print("ETL process completed successfully!")

if __name__ == "__main__":
    main()
