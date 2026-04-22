# from pyspark.sql import SparkSession, Row
# from pyspark.sql.functions import col, avg, desc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import random
import urllib.request
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INPUT_CSV = os.path.join(DATA_DIR, "job_vec.csv")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "clustered_output")
SKILL_WEIGHT = 1.0
EXP_WEIGHT = 8.0
OTHER_WEIGHT = 0.4
LLM_API_URL = os.environ.get("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "your deepseep api key")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
MAX_LLM_WORKERS = int(os.environ.get("MAX_LLM_WORKERS", "6"))
LLM_CACHE_FILE = os.path.join(OUTPUT_FOLDER, "llm_cache.json")
# # MySQL connection properties
# mysql_properties = {
#     "user": "root",
#     "password": "password",
#     "driver": "com.mysql.cj.jdbc.Driver"
# }

# # Initialize Spark session with Hive support for ETL outputs
# def create_spark_session():
#     return SparkSession.builder \
#         .appName("CareerMind Statistics Analysis") \
#         .enableHiveSupport() \
#         .config("spark.jars.packages", "mysql:mysql-connector-java:8.0.33") \
#         .getOrCreate()

# Read ETL outputs from Hive as input to statistics analysis
def read_etl_outputs(spark):
    # job_df = spark.table("job_feature_table")
    # user_df = spark.table("user_feature_table")
    # interaction_df = spark.table("interaction_table")
    # static_source_df = spark.table("report_static_source")
    # return job_df, user_df, interaction_df, static_source_df
    return pd.read_csv(INPUT_CSV)

# # Build the analytics result table from ETL output tables
# def build_statistics_table(spark, job_df, user_df, interaction_df, static_source_df):
#     total_jobs = job_df.count()
#     total_users = user_df.count()
#     total_interactions = interaction_df.count()
#     avg_salary_max = job_df.agg(avg("salary_max")).first()[0] or 0.0
#     avg_salary_min = job_df.agg(avg("salary_min")).first()[0] or 0.0

#     top_location_row = job_df.groupBy("location").count().orderBy(desc("count")).limit(1).collect()
#     top_location = top_location_row[0]["location"] if top_location_row else "N/A"

#     top_industry_row = job_df.groupBy("industry").count().orderBy(desc("count")).limit(1).collect()
#     top_industry = top_industry_row[0]["industry"] if top_industry_row else "N/A"

#     static_source_count = static_source_df.count()

#     now = datetime.now()
#     statistics = [
#         Row(stat_key="totalJobs", stat_name="Total Jobs", metric_category="job", dimension="all", metric_value=float(total_jobs), metric_text=None, metric_time=now, source_table="job_feature_table", created_at=now, updated_at=now),
#         Row(stat_key="totalUsers", stat_name="Total Users", metric_category="user", dimension="all", metric_value=float(total_users), metric_text=None, metric_time=now, source_table="user_feature_table", created_at=now, updated_at=now),
#         Row(stat_key="totalInteractions", stat_name="Total Interactions", metric_category="interaction", dimension="all", metric_value=float(total_interactions), metric_text=None, metric_time=now, source_table="interaction_table", created_at=now, updated_at=now),
#         Row(stat_key="avgSalaryMax", stat_name="Average Max Salary", metric_category="job", dimension="all", metric_value=float(avg_salary_max), metric_text=None, metric_time=now, source_table="job_feature_table", created_at=now, updated_at=now),
#         Row(stat_key="avgSalaryMin", stat_name="Average Min Salary", metric_category="job", dimension="all", metric_value=float(avg_salary_min), metric_text=None, metric_time=now, source_table="job_feature_table", created_at=now, updated_at=now),
#         Row(stat_key="topLocation", stat_name="Top Job Location", metric_category="job", dimension="location", metric_value=None, metric_text=top_location, metric_time=now, source_table="job_feature_table", created_at=now, updated_at=now),
#         Row(stat_key="topIndustry", stat_name="Top Job Industry", metric_category="job", dimension="industry", metric_value=None, metric_text=top_industry, metric_time=now, source_table="job_feature_table", created_at=now, updated_at=now),
#         Row(stat_key="staticSourceRecords", stat_name="Static Report Source Records", metric_category="static", dimension="all", metric_value=float(static_source_count), metric_text=None, metric_time=now, source_table="report_static_source", created_at=now, updated_at=now)
#     ]

#     return spark.createDataFrame(statistics)

# # Write analysis results to MySQL report_stats
# def write_statistics_to_mysql(statistics_df):
#     statistics_df.write.jdbc(
#         url="jdbc:mysql://mysql:3306/careermind",
#         table="report_stats",
#         mode="overwrite",
#         properties=mysql_properties
#     )

def silhouette_method(X, max_k=10):
    n_samples = len(X)
    max_k = min(max_k, n_samples - 1)  # Ensure max_k < n_samples
    if max_k < 2:
        return 2  # Default to 2 clusters if not enough samples
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    # plt.plot(range(2, max_k+1), silhouette_scores, 'bo-')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Method')
    # plt.show()
    best_k = silhouette_scores.index(max(silhouette_scores)) + 2
    return best_k

def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).replace("\n", "；").replace("\r", "；").strip()


def normalize_segment(segment):
    norms = np.linalg.norm(segment, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return segment / norms


def build_weighted_features(X):
    total_dim = X.shape[1]
    if total_dim < 3:
        raise ValueError(f"Vector dimension too small: {total_dim}")
    exp_dim = 1
    remaining = total_dim - exp_dim
    skill_dim = remaining // 2
    other_dim = remaining - skill_dim
    
    exp = X[:, :exp_dim].astype(float)
    skill = X[:, exp_dim:exp_dim + skill_dim].astype(float)
    other = X[:, exp_dim + skill_dim:].astype(float)

    exp = np.clip(exp, 0.0, 1.0)
    skill = normalize_segment(skill)
    other = normalize_segment(other)

    weighted_skill = skill * SKILL_WEIGHT
    weighted_exp = exp * EXP_WEIGHT
    weighted_other = other * OTHER_WEIGHT

    return np.hstack([weighted_skill, weighted_exp, weighted_other])


def extract_requirement_phrases(texts, top_n=5, min_len=10):
    joined = "；".join([normalize_text(text) for text in texts if str(text).strip()])
    if not joined:
        return []
    separators = "；;。.!?,，、\n"
    for sep in separators:
        joined = joined.replace(sep, "；")
    segments = [seg.strip() for seg in joined.split("；") if seg.strip()]
    segments = [seg for seg in segments if len(seg) >= min_len]
    if not segments:
        return []
    counts = pd.Series(segments).value_counts()
    return counts.head(top_n).index.tolist()


def sample_other_requirements(cluster_df, sample_size=30, random_state=42):
    texts = cluster_df['other_requirement'].dropna().astype(str)
    if len(texts) == 0:
        return []
    if len(texts) <= sample_size:
        return texts.tolist()
    return texts.sample(n=sample_size, random_state=random_state).tolist()


def build_llm_prompt(other_texts, industry, cluster_id):
    prompt_lines = [
        "以下是属于同一类岗位的任职要求，请总结出 3-5 条核心共性。",
        f"行业：{industry}",
        f"簇编号：{cluster_id}",
        "请重点提炼职责/资质/能力/经验/态度等方面的共同要求。",
        "请总结这个行业里这个岗位的共性，避免总结特定公司招聘的细节。",
        "如果某些内容不适合作为共性，可以忽略它们。",
        "以下是采样的任职要求："
    ]
    for idx, text in enumerate(other_texts, start=1):
        prompt_lines.append(f"{idx}. {normalize_text(text)}")
    prompt_lines.append("\n请输出 3-5 条简洁的核心共性，每条不宜太长。")
    return "\n".join(prompt_lines)


def build_profile_name_prompt(job_titles, core_skills, industry, cluster_id):
    prompt_lines = [
        "请为以下岗位信息生成一个简洁的岗位名称（10字以内）。",
        f"行业：{industry}",
        f"簇编号：{cluster_id}",
        "请总结这个行业里这个岗位的共性，避免特定公司招聘的细节。",
        "岗位信息：",
        f"常见职位标题：{', '.join(job_titles[:5])}",  # 取前5个标题
        f"核心技能：{', '.join(core_skills[:10])}",  # 取前10个技能
        "",
        "请输出一个简洁的岗位名称，直接输出名称即可，不要其他解释。"
    ]
    return "\n".join(prompt_lines)


def build_cluster_insight_prompt(profile_name_prompt, requirement_prompt):
    return "\n\n".join([
        "请基于以下两段任务描述，一次性返回 JSON 结果。",
        "必须输出 JSON 对象，格式为：",
        '{"profile_name":"岗位名称","other_requirements":["要求1","要求2","要求3"]}',
        "注意：profile_name 不超过10个中文字符；other_requirements 输出3-5条。",
        "任务一（岗位命名）：",
        profile_name_prompt,
        "任务二（任职要求共性总结）：",
        requirement_prompt
    ])


def load_llm_cache():
    if not os.path.exists(LLM_CACHE_FILE):
        return {}
    try:
        with open(LLM_CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_llm_cache(cache):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(LLM_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def parse_json_from_llm(content):
    if not content:
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return None
    return None


def call_llm(prompt_text, system_text, temperature=0.2):
    """Call OpenAI-compatible chat completion endpoint and return text content."""
    if not LLM_API_KEY:
        return None
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_text
            },
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "temperature": temperature
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        LLM_API_URL,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLM_API_KEY}"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)
        content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content if content else None
    except Exception as exc:
        print(f"LLM request failed: {exc}")
        return None


def generate_cluster_insights_with_llm(profile_name_prompt, requirement_prompt):
    combined_prompt = build_cluster_insight_prompt(profile_name_prompt, requirement_prompt)
    content = call_llm(
        combined_prompt,
        "你是招聘分析助手。严格返回 JSON 对象，不要输出任何额外文字。",
        temperature=0.2
    )
    parsed = parse_json_from_llm(content)
    if not isinstance(parsed, dict):
        return None, []
    profile_name = str(parsed.get("profile_name", "")).strip()[:20]
    other_reqs = parsed.get("other_requirements", [])
    if not isinstance(other_reqs, list):
        other_reqs = []
    cleaned = [str(item).strip() for item in other_reqs if str(item).strip()]
    return profile_name if profile_name else None, cleaned[:5]


def silhouette_method_and_cluster(X, max_k=8):
    # 先确定 best_k
    best_k = silhouette_method(X, max_k)   # 上面那个函数（会画图）
    # 用 best_k 重新训练并返回结果
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = final_kmeans.fit_predict(X)
    return best_k, labels, final_kmeans
# Main analysis process
def main():
    # spark = create_spark_session()
    # try:
    #     job_df, user_df, interaction_df, static_source_df = read_etl_outputs(spark)
    #     print("Loaded ETL outputs from Hive.")

    #     statistics_df = build_statistics_table(spark, job_df, user_df, interaction_df, static_source_df)
    #     statistics_df.write.mode("overwrite").saveAsTable("statistics_table")
    #     write_statistics_to_mysql(statistics_df)

    #     print("Statistics analysis complete: results saved to Hive and MySQL.")
    # finally:
    #     spark.stop()

    # Load job vectors
    df = read_etl_outputs(None)
    print(f"Loaded {len(df)} job records with vectors.")

    # Get vector columns
    vec_cols = [col for col in df.columns if col.startswith('job_vec_')]
    print(f"Vector dimensions: {len(vec_cols)}")

    # Group by industry using industry_group categories
    industries = df['industry_group'].unique()
    print(f"Industries: {len(industries)} categories found")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    profile_rows = []

    for industry in industries:
        industry_df = df[df['industry_group'] == industry].copy()
        n_jobs = len(industry_df)
        if n_jobs < 10:
            print(f"Skipping industry: not enough data ({n_jobs} jobs)")
            continue
        
        print(f"Clustering industry: {n_jobs} jobs")
        
        X = industry_df[vec_cols].values
        weighted_X = build_weighted_features(X)
        
        # Use silhouette method to determine k and cluster
        best_k, labels, kmeans = silhouette_method_and_cluster(weighted_X, max_k=min(10, n_jobs-1))
        labels = labels + 1
        print(f"Best k for industry: {best_k}")
        
        # Optionally save results
        industry_df['cluster'] = labels
        # Remove vector columns from output
        output_df = industry_df.drop(columns=vec_cols)
        output_df.to_csv(f"{OUTPUT_FOLDER}/clustered_{industry.replace('/', '_').replace(' ', '_')}.csv", index=False)
        print(f"Saved clustered data for industry")
        
        # Analyze each cluster
        for cluster_id in range(1, best_k + 1):
            cluster_df = industry_df[industry_df['cluster'] == cluster_id]
            if len(cluster_df) == 0:
                continue
            
            # Job titles
            job_titles = cluster_df['job_title'].value_counts().head(5).index.tolist()
            most_common_title = job_titles[0] if job_titles else "N/A"
            
            # Core skills
            skill_col = 'merged_job_skills' if 'merged_job_skills' in cluster_df.columns else 'job_skills'
            all_skills = cluster_df[skill_col].fillna("").str.split(r'[,;；]+').explode().str.strip()
            core_skills = all_skills[all_skills != ""].value_counts().head(10).index.tolist()
            
            # Experience requirements
            exp_counts = cluster_df['experience_required'].fillna("未知").value_counts()
            most_common_exp = exp_counts.index[0] if not exp_counts.empty else "N/A"
            
            # Other requirements from other_requirement only
            requirement_texts = cluster_df['other_requirement'].dropna().astype(str).tolist()
            other_reqs_fallback = extract_requirement_phrases(requirement_texts, top_n=5)
            sampled_texts = sample_other_requirements(cluster_df, sample_size=30)
            prompt_text = build_llm_prompt(sampled_texts, industry, cluster_id)
            prompt_file = f"{OUTPUT_FOLDER}/prompt_{industry.replace('/', '_').replace(' ', '_')}_{cluster_id}.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt_text)
            # Generate profile name using LLM
            profile_name_prompt = build_profile_name_prompt(job_titles, core_skills, industry, cluster_id)
            profile_name_file = f"{OUTPUT_FOLDER}/profile_name_prompt_{industry.replace('/', '_').replace(' ', '_')}_{cluster_id}.txt"
            with open(profile_name_file, 'w', encoding='utf-8') as f:
                f.write(profile_name_prompt)
            
            # Salary stats: 平均最低薪资和最高薪资（基于聚类内所有职位）
            # 使用 ETL 中标准化的字段（元/月），避免混合不同单位的原始值
            avg_min_salary = cluster_df['salary_min_norm'].astype(float).mean()
            avg_max_salary = cluster_df['salary_max_norm'].astype(float).mean()
            
            # Education: 聚类内最常见的学历要求
            education_counts = cluster_df['education_required'].fillna("未知").value_counts()
            most_common_edu = education_counts.index[0] if not education_counts.empty else "N/A"
            
            # Calculate cluster percentage in this industry
            cluster_percentage = round(100 * len(cluster_df) / n_jobs, 2)
            
            print(f"\n--- Profile: Cluster {cluster_id} ---")
            print(f"Number of jobs: {len(cluster_df)} ({cluster_percentage}% of industry)")
            print(f"Common job titles: {len(job_titles)} titles")
            print(f"Core skills: {len(core_skills)} skills")
            print(f"Experience requirement: {len(str(most_common_exp))} chars")
            print(f"Other requirements: {len(other_reqs_fallback)} phrases")
            print(f"Prompt file generated")
            print(f"Average salary range: {avg_min_salary:.2f} - {avg_max_salary:.2f}")
            print(f"Education requirement: {len(str(most_common_edu))} chars")

            profile_rows.append({
                "industry_group": industry,
                "cluster_id": cluster_id,
                "profile_name": f"Cluster {cluster_id} ({industry})",
                "profile_name_prompt_file": profile_name_file,
                "job_count": len(cluster_df),
                "cluster_percentage_in_industry": cluster_percentage,
                "top_titles": "; ".join(job_titles),
                "core_skills": "; ".join(core_skills),
                "experience": most_common_exp,
                "other_requirements": "；".join(other_reqs_fallback),
                "education": most_common_edu,
                "salary_min_avg": round(avg_min_salary, 2),
                "salary_max_avg": round(avg_max_salary, 2),
                "llm_prompt_file": prompt_file,
                "_profile_name_prompt": profile_name_prompt,
                "_requirements_prompt": prompt_text
            })

    if profile_rows and LLM_API_KEY:
        llm_cache = load_llm_cache()

        def enrich_row(row):
            cache_key_text = row["_profile_name_prompt"] + "\n\n" + row["_requirements_prompt"]
            cache_key = hashlib.sha256(cache_key_text.encode("utf-8")).hexdigest()
            if cache_key in llm_cache:
                cached = llm_cache[cache_key]
                return row, cached.get("profile_name"), cached.get("other_requirements", [])
            profile_name, other_reqs = generate_cluster_insights_with_llm(
                row["_profile_name_prompt"],
                row["_requirements_prompt"]
            )
            llm_cache[cache_key] = {
                "profile_name": profile_name,
                "other_requirements": other_reqs
            }
            return row, profile_name, other_reqs

        with ThreadPoolExecutor(max_workers=max(1, MAX_LLM_WORKERS)) as executor:
            futures = [executor.submit(enrich_row, row) for row in profile_rows]
            for future in as_completed(futures):
                row, profile_name, other_reqs = future.result()
                if profile_name:
                    row["profile_name"] = profile_name
                if other_reqs:
                    row["other_requirements"] = "；".join(other_reqs)
        save_llm_cache(llm_cache)

    if profile_rows:
        for row in profile_rows:
            row.pop("_profile_name_prompt", None)
            row.pop("_requirements_prompt", None)
        summary_df = pd.DataFrame(profile_rows)
        summary_df.to_csv(f"{OUTPUT_FOLDER}/cluster_profiles.csv", index=False)
        print(f"Saved cluster profile summary to {OUTPUT_FOLDER}/cluster_profiles.csv")

    print("Clustering analysis complete.")


if __name__ == "__main__":
    main()
