"""data_bridge.py

Lightweight data interface for the CareerMind Streamlit dashboard.

Purpose
- Provide simple, well-documented functions that the Streamlit app calls
  to obtain cleaned job tables, cluster summaries, skill-importance and
  experience / salary projection curves.

Design notes
- This module intentionally does NOT run heavy ML pipelines. It looks for
  preprocessed CSVs in a `data/` folder (relative to this file). If those
  are missing it will try fallbacks in the parent workspace (e.g. the
  repository's `jobs(1).csv` or `clustered_output/` files).

Expected files (put these under `career_mind_dashboard/data/` for the
dashboard to use them directly):

- jobs_clean.csv
  columns: job_id, title, industry, salary_min, salary_max, salary_avg,
           salary_unit, salary_period, education_required, other_requirement,
           job_skills, city, province, lat, lon, cluster_id

- cluster_results.csv  (or cluster_results_{industry}.csv)
  columns: cluster_id, industry, core_skills, salary_min_avg, salary_max_avg,
           count, sample_size

- skill_importance.csv
  columns: cluster_id, skill, importance

- exp_curve.csv
  columns: cluster_id, year, salary

- report_sections.json
  keys: discussion, limitations, conclusion, appendix_references, appendix_methodology

If any of these are missing the functions below will either return a
reasonable empty DataFrame or a small synthetic example so the UI still
renders. Put your actual processed outputs in `career_mind_dashboard/data/`.
"""

import os
import re
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd


_HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(_HERE, "data")
PARENT_DIR = os.path.abspath(os.path.join(_HERE, ".."))


def _find(path_candidates):
    for p in path_candidates:
        if os.path.exists(p):
            return p
    return None


def _is_under(dir_path: str, file_path: Optional[str]) -> bool:
    if not file_path or not os.path.isfile(file_path):
        return False
    try:
        d = os.path.realpath(dir_path)
        f = os.path.realpath(file_path)
        return os.path.commonpath([d, f]) == d
    except (ValueError, OSError):
        return False


def _source_bucket(file_path: Optional[str]) -> str:
    """Human-readable origin folder for sidebar (Chinese)."""
    if not file_path:
        return "未找到"
    if _is_under(DATA_DIR, file_path):
        return "data/（标准）"
    sub = os.path.join(PARENT_DIR, "clustered_output")
    if _is_under(sub, file_path):
        return "clustered_output/（回退）"
    sub = os.path.join(PARENT_DIR, "regression_output")
    if _is_under(sub, file_path):
        return "regression_output/（回退）"
    if _is_under(PARENT_DIR, file_path):
        return "项目根目录 analysis/（回退）"
    return "其他路径"


def _short_display_path(file_path: Optional[str]) -> str:
    if not file_path:
        return "—"
    try:
        rel = os.path.relpath(file_path, PARENT_DIR)
    except ValueError:
        rel = file_path
    return rel.replace("\\", "/")


def _read_csv_anywhere(filename: str) -> Optional[pd.DataFrame]:
    """Search for `filename` in `data/`, then parent folder and clustered_output.

    Returns a DataFrame or None.
    """
    candidates = [os.path.join(DATA_DIR, filename), os.path.join(PARENT_DIR, filename)]
    # also look inside clustered_output if present in parent
    candidates.append(os.path.join(PARENT_DIR, "clustered_output", filename))
    path = _find(candidates)
    if path:
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None
    return None


def _resolve_jobs_csv_path() -> Optional[str]:
    """Same resolution order as load_jobs(): jobs_clean then job_vec / jobs fallbacks."""
    for name in ("jobs_clean.csv", "job_vec.csv", "jobs(1).csv", "jobs.csv"):
        p = _find(
            [
                os.path.join(DATA_DIR, name),
                os.path.join(PARENT_DIR, name),
                os.path.join(PARENT_DIR, "clustered_output", name),
            ]
        )
        if p:
            return p
    return None


def _resolve_cluster_profiles_path() -> Optional[str]:
    return _find(
        [
            os.path.join(DATA_DIR, "cluster_profiles.csv"),
            os.path.join(PARENT_DIR, "clustered_output", "cluster_profiles.csv"),
        ]
    )


def _resolve_skill_value_robust_path() -> Optional[str]:
    return _find(
        [
            os.path.join(DATA_DIR, "skill_value_robust.csv"),
            os.path.join(PARENT_DIR, "regression_output", "skill_value_robust.csv"),
        ]
    )


def _resolve_exp_curve_csv_path() -> Optional[str]:
    return _find(
        [
            os.path.join(DATA_DIR, "exp_curve.csv"),
            os.path.join(PARENT_DIR, "exp_curve.csv"),
            os.path.join(PARENT_DIR, "regression_output", "exp_curve.csv"),
        ]
    )


def _resolve_skill_impact_path() -> Optional[str]:
    """回归管线输出的 MI/系数技能表（大样本行业），与 skill_value_robust（中小样本）互补。"""
    return _find(
        [
            os.path.join(DATA_DIR, "skill_impact.csv"),
            os.path.join(PARENT_DIR, "regression_output", "skill_impact.csv"),
            os.path.join(PARENT_DIR, "skill_impact.csv"),
        ]
    )


def _strip_industry(s: Optional[str]) -> str:
    return (s or "").strip()


def _eq_cluster_id(series: pd.Series, cluster_id) -> pd.Series:
    """匹配 cluster_id（CSV 中常为整数，界面可能为 float）。"""
    if cluster_id is None or pd.isna(cluster_id):
        return pd.Series(False, index=series.index)
    try:
        target = int(float(cluster_id))
    except (TypeError, ValueError):
        target = cluster_id
    try:
        s_int = pd.to_numeric(series, errors="coerce")
        return s_int == target
    except Exception:
        return series.astype(str) == str(cluster_id)


def _resolve_skill_merge_preview_path() -> Optional[str]:
    return _find(
        [
            os.path.join(DATA_DIR, "skill_merge_preview.csv"),
            os.path.join(PARENT_DIR, "skill_merge_preview.csv"),
        ]
    )


# 与 etl.SKILL_MERGE_CORRELATION_THRESHOLD 保持一致（避免 import etl 触发重型依赖）
SKILL_MERGE_CORRELATION_THRESHOLD_DOC = 0.75


def load_skill_merge_preview() -> pd.DataFrame:
    """技能标签合并预览表：行业、建议合并标签、原始技能列表（分号分隔）。"""
    path = _resolve_skill_merge_preview_path()
    if not path:
        return pd.DataFrame(columns=["industry_group", "suggested_label", "original_skills"])
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=["industry_group", "suggested_label", "original_skills"])


def _parse_salary_str(s: str):
    """Parse a salary string and return (min, max, avg_monthly).

    Handles common patterns like '10k-15k/月', '1-2万/年', '面议' (ignored).
    Returns (None, None, None) when parsing fails.
    """
    if pd.isna(s):
        return (None, None, None)
    s0 = str(s).lower().replace('\u00a0', ' ').strip()
    if '面议' in s0 or 'negotiable' in s0:
        return (None, None, None)

    # detect period
    period = 'month'
    if '年' in s0 or '/year' in s0 or '/年' in s0:
        period = 'year'

    # numbers
    nums = re.findall(r"(\d+(?:\.\d+)?)", s0)
    if not nums:
        return (None, None, None)
    try:
        nums = [float(x) for x in nums]
    except Exception:
        return (None, None, None)

    multiplier = 1.0
    if 'k' in s0:
        multiplier = 1000.0
    if '万' in s0:
        multiplier = 10000.0

    if len(nums) >= 2:
        a, b = nums[0] * multiplier, nums[1] * multiplier
    else:
        a = b = nums[0] * multiplier

    avg = (a + b) / 2.0
    # convert yearly to monthly for consistency
    if period == 'year':
        avg = avg / 12.0
        a = a / 12.0
        b = b / 12.0

    return (a, b, avg)


def _ensure_latlon(df: pd.DataFrame) -> pd.DataFrame:
    # normalize common lat/lon column names to 'lat'/'lon'
    if 'lat' not in df.columns:
        for c in ('latitude', 'latd', 'latitudes'):
            if c in df.columns:
                df['lat'] = pd.to_numeric(df[c], errors='coerce')
                break
    if 'lon' not in df.columns:
        for c in ('longitude', 'lng', 'lonng'):
            if c in df.columns:
                df['lon'] = pd.to_numeric(df[c], errors='coerce')
                break
    return df


def load_jobs() -> pd.DataFrame:
    """Load jobs from job_vec.csv (ETL output) with column normalization.
    
    Reads from parent/job_vec.csv which has standardized salary_avg and other fields.
    Normalizes column names to: job_id, title, industry, salary_avg, education_required, etc.
    """
    df = _read_csv_anywhere('jobs_clean.csv')
    if df is None:
        # Prefer job_vec.csv (ETL output with standardized fields)
        for fallback in ['job_vec.csv', 'jobs(1).csv', 'jobs.csv']:
            df = _read_csv_anywhere(fallback)
            if df is not None:
                break
    
    if df is None:
        cols = ['job_id', 'title', 'industry', 'salary_avg', 'education_required', 
                'other_requirement', 'job_skills', 'city', 'province', 'lat', 'lon']
        return pd.DataFrame(columns=cols)
    
    # Normalize column names from various sources
    renames = {
        'job_title': 'title',
        'avg_salary': 'salary_avg'
    }
    df = df.rename(columns=renames)
    
    # Always use cleaned industry grouping first.
    if 'industry_group' in df.columns:
        df['industry'] = df['industry_group']
    elif 'industry' not in df.columns and 'industry_name' in df.columns:
        # Last-resort fallback for older datasets.
        df['industry'] = df['industry_name']
    
    # Remove duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # Ensure salary_avg is numeric (ETL already standardized to RMB/month)
    if 'salary_avg' in df.columns:
        df['salary_avg'] = pd.to_numeric(df['salary_avg'], errors='coerce')
    else:
        df['salary_avg'] = np.nan
    
    df = _ensure_latlon(df)
    
    # Ensure required columns exist
    for c in ['education_required', 'other_requirement', 'job_skills', 'industry']:
        if c not in df.columns:
            df[c] = ""
    
    return df


def extract_education_labels(df: pd.DataFrame) -> pd.Series:
    """Create a compact education label from education_required and other_requirement.

    Returns a pandas Series of labels like: '博士','硕士','本科','大专','中专/高中','不限/其他'
    """
    def pick_label(row):
        text = ' '.join([str(row.get('education_required', '') or ''), str(row.get('other_requirement', '') or '')])
        text = text.lower()
        if any(k in text for k in ['博士', 'phd']):
            return '博士'
        if any(k in text for k in ['硕士', 'master', '研究生']):
            return '硕士'
        if any(k in text for k in ['本科', '学士']):
            return '本科'
        if any(k in text for k in ['大专', '专科']):
            return '大专'
        if any(k in text for k in ['中专', '高中']):
            return '中专/高中'
        if any(k in text for k in ['不限', 'no requirement', '无要求']):
            return '不限/其他'
        return '不限/其他'

    return df.apply(pick_label, axis=1)


def get_overview_stats(jobs: pd.DataFrame) -> Dict:
    jobs = jobs.copy()
    total = int(len(jobs))
    avg_salary = float(jobs['salary_avg'].dropna().mean()) if not jobs['salary_avg'].dropna().empty else None
    industries = sorted(jobs['industry'].dropna().unique().tolist())
    top_industries = jobs.groupby('industry').size().sort_values(ascending=False).head(10)
    return {
        'total_jobs': total,
        'avg_salary': avg_salary,
        'industry_count': len(industries),
        'top_industries': top_industries.reset_index().rename(columns={0: 'count'})
    }


def get_industries(jobs: pd.DataFrame):
    # Normalize column name variants
    industry_col = 'industry' if 'industry' in jobs.columns else 'industry_group' if 'industry_group' in jobs.columns else 'industry_name'
    if industry_col not in jobs.columns:
        return []
    return sorted(jobs[industry_col].dropna().unique().tolist())


def _pinyin_sort_key(s: str) -> str:
    """全拼小写，用于中文行业名按拼音字母序排序；无 pypinyin 时回退为原字符串小写。"""
    s = str(s)
    try:
        from pypinyin import lazy_pinyin  # type: ignore

        return "".join(lazy_pinyin(s)).lower()
    except Exception:
        return s.lower()


def get_industries_from_cluster_profiles() -> List[str]:
    """行业穿透可选行业：仅包含 cluster_profiles（data/ 优先）中已聚类/画像的行业组，按拼音排序。"""
    path = _resolve_cluster_profiles_path()
    if not path:
        return []
    try:
        cdf = pd.read_csv(path, low_memory=False)
    except Exception:
        return []
    if "industry_group" in cdf.columns:
        col = "industry_group"
    elif "industry" in cdf.columns:
        col = "industry"
    else:
        return []
    ser = cdf[col].astype(str).str.strip()
    ser = ser[(ser != "") & (ser.str.lower() != "nan")]
    unique = sorted(set(ser.tolist()), key=_pinyin_sort_key)
    return unique


def get_clusters_for_industry(industry: str) -> pd.DataFrame:
    """Return cluster summaries for an industry from clustered_output/cluster_profiles.csv.

    仅返回 industry_group 与所选行业**完全一致**的行；不再使用 str.contains，避免误匹配；
    筛选为空时返回空表（绝不返回整张 cluster_profiles，否则会混入其他行业聚类）。
    """
    cols = ['cluster_id', 'industry', 'core_skills', 'salary_min_avg', 'salary_max_avg', 'count', 'sample_size']
    ind = _strip_industry(industry)
    if not ind:
        return pd.DataFrame(columns=cols)

    general = _read_csv_anywhere('cluster_results.csv')
    if general is not None and ('industry' in general.columns or 'industry_group' in general.columns):
        try:
            industry_col = 'industry' if 'industry' in general.columns else 'industry_group'
            ig = general[industry_col].astype(str).str.strip()
            rows = general[ig == ind].copy()
            if not rows.empty:
                return rows
        except Exception:
            pass

    path = _resolve_cluster_profiles_path()
    if not path:
        return pd.DataFrame(columns=cols)
    try:
        cdf = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame(columns=cols)

    if 'industry_group' not in cdf.columns:
        return pd.DataFrame(columns=cols)

    ig = cdf['industry_group'].astype(str).str.strip()
    rows = cdf[ig == ind].copy()
    if rows.empty:
        return pd.DataFrame(columns=cols)

    renames = {'industry_group': 'industry', 'job_count': 'count'}
    rows = rows.rename(columns=renames)
    if 'count' not in rows.columns:
        rows['count'] = 0
    rows['sample_size'] = rows['count']
    if 'cluster_id' in rows.columns:
        rows = rows.sort_values('cluster_id', kind='stable')
    return rows


def get_skill_importance(cluster_id, industry: str = None) -> pd.DataFrame:
    """Load skill importance for UI.

    优先级：适配层 skill_importance.csv（按 cluster）→ regression skill_impact.csv（按行业精确匹配）
    → skill_value_robust.csv（中小样本行业）→ 占位示例。

    当前回归产物多为**行业级**（无 cluster 维度），cluster_id 仅用于读取适配表。
    Returns columns: skill, importance（importance 归一化到 0–1）。
    """
    ind = _strip_industry(industry)

    df = _read_csv_anywhere('skill_importance.csv')
    if df is not None and 'cluster_id' in df.columns and cluster_id is not None:
        mask = _eq_cluster_id(df['cluster_id'], cluster_id)
        result = df.loc[mask].copy()
        if not result.empty and 'skill' in result.columns and 'importance' in result.columns:
            return result[['skill', 'importance']].copy()

    if ind:
        ip = _resolve_skill_impact_path()
        if ip:
            try:
                df = pd.read_csv(ip, low_memory=False)
                if 'industry_group' in df.columns and 'skill' in df.columns and 'coefficient' in df.columns:
                    ig = df['industry_group'].astype(str).str.strip()
                    rows = df.loc[ig == ind].copy()
                    if not rows.empty:
                        rows['importance'] = pd.to_numeric(rows['coefficient'], errors='coerce').abs()
                        mx = float(rows['importance'].max())
                        if mx > 0:
                            rows['importance'] = rows['importance'] / mx
                        rows = rows.sort_values('importance', ascending=False).head(20)
                        return rows[['skill', 'importance']].copy()
            except Exception:
                pass

        skill_robust_path = _resolve_skill_value_robust_path()
        if skill_robust_path:
            try:
                df = pd.read_csv(skill_robust_path, low_memory=False)
                if 'industry_group' in df.columns and 'skill' in df.columns and 'pure_skill_value' in df.columns:
                    ig = df['industry_group'].astype(str).str.strip()
                    rows = df.loc[ig == ind].copy()
                    if not rows.empty:
                        rows['importance'] = pd.to_numeric(rows['pure_skill_value'], errors='coerce').abs()
                        mx = float(rows['importance'].max())
                        if mx > 0:
                            rows['importance'] = rows['importance'] / mx
                        rows = rows.sort_values('importance', ascending=False).head(20)
                        return rows[['skill', 'importance']].copy()
            except Exception:
                pass

    return pd.DataFrame(columns=['skill', 'importance'])


def get_exp_curve(cluster_id, industry: str = None) -> pd.DataFrame:
    """经验–薪资曲线：行业级（regression_output/exp_curve.csv），按 industry_group **精确**匹配。

    若有适配层 exp_curve.csv 且含 cluster_id，则优先按 cluster 过滤。返回 year, salary。
    """
    ind = _strip_industry(industry)

    def _ensure_0_to_10_years(curve_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure experience curve contains every point from 0 to 10 years."""
        if curve_df is None or curve_df.empty or 'salary' not in curve_df.columns:
            return pd.DataFrame(columns=['year', 'salary'])
        d = curve_df.copy()
        if 'exp_years' in d.columns:
            d['exp_years'] = pd.to_numeric(d['exp_years'], errors='coerce')
        elif 'years_experience' in d.columns:
            d['exp_years'] = pd.to_numeric(d['years_experience'], errors='coerce')
        elif 'year' in d.columns:
            d['exp_years'] = pd.to_numeric(d['year'], errors='coerce') - 2023
        else:
            return pd.DataFrame(columns=['year', 'salary'])

        d['salary'] = pd.to_numeric(d['salary'], errors='coerce')
        d = d.dropna(subset=['exp_years', 'salary'])
        if d.empty:
            return pd.DataFrame(columns=['year', 'salary'])

        # Aggregate duplicated years and linearly interpolate missing years.
        s = d.groupby(d['exp_years'].round().astype(int), as_index=True)['salary'].mean().sort_index()
        full_idx = pd.Index(range(0, 11), name='exp_years')
        s = s.reindex(full_idx).interpolate(method='linear', limit_direction='both')
        out = pd.DataFrame({'year': 2023 + s.index.astype(int), 'salary': s.values})
        return out[['year', 'salary']]

    df = _read_csv_anywhere('exp_curve.csv')
    if df is not None and 'cluster_id' in df.columns and cluster_id is not None:
        mask = _eq_cluster_id(df['cluster_id'], cluster_id)
        result = df.loc[mask].copy()
        if not result.empty:
            if 'year' not in result.columns and 'years_experience' in result.columns:
                result['year'] = 2023 + pd.to_numeric(result['years_experience'], errors='coerce').fillna(0).astype(int)
            if 'salary' not in result.columns and 'predicted_salary' in result.columns:
                result = result.rename(columns={'predicted_salary': 'salary'})
            if {'year', 'salary'}.issubset(result.columns):
                return _ensure_0_to_10_years(result[['year', 'salary']].sort_values('year').copy())

    if ind:
        exp_path = _resolve_exp_curve_csv_path()
        if exp_path:
            try:
                df = pd.read_csv(exp_path, low_memory=False)
                if 'industry_group' in df.columns and 'years_experience' in df.columns:
                    ig = df['industry_group'].astype(str).str.strip()
                    rows = df.loc[ig == ind].copy()
                    if not rows.empty:
                        rows['year'] = 2023 + pd.to_numeric(rows['years_experience'], errors='coerce').fillna(0).astype(int)
                        rows = rows.rename(columns={'predicted_salary': 'salary'})
                        out = rows[['year', 'salary']].sort_values('year').copy()
                        return _ensure_0_to_10_years(out)
            except Exception:
                pass

    years = list(range(2023, 2034))
    base = 8000 + (abs(hash(str(cluster_id) + '|' + ind)) % 5000)
    salaries = [base * (1 + 0.03) ** (y - 2023) for y in years]
    return pd.DataFrame({'year': years, 'salary': salaries})


def load_report_sections() -> Dict[str, str]:
    """Load discussion / limitations / conclusion from report_sections.json if present.

    Returns a dict with keys: discussion, limitations, conclusion, appendix_references,
    appendix_methodology, appendix_skill_merge_intro（若存在 skill_merge_preview 则自动补充说明）
    """
    path = _find([os.path.join(DATA_DIR, 'report_sections.json'), os.path.join(PARENT_DIR, 'report_sections.json')])
    sections: Dict[str, str] = {}
    if path:
        try:
            sections = pd.read_json(path, typ='series').to_dict()
        except Exception:
            try:
                import json
                with open(path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                    sections = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
            except Exception:
                sections = {}

    defaults = {
        'discussion': '',
        'limitations': '',
        'conclusion': '',
        'appendix_references': '',
        'appendix_methodology': '',
        'appendix_skill_merge_intro': '',
    }
    for k, v in defaults.items():
        sections.setdefault(k, v)

    merge_path = _resolve_skill_merge_preview_path()
    intro_auto = (
        f"技能标签合并：在 `transform_jobs_to_vector_table` 中调用 `compute_skill_merge_preview`，"
        f"按行业分组构建技能 0/1 共现矩阵，对技能列计算 Pearson 相关系数；当 r > {SKILL_MERGE_CORRELATION_THRESHOLD_DOC} "
        f"（与 `etl.SKILL_MERGE_CORRELATION_THRESHOLD` 一致）时在并查集上合并为同一技能组，"
        f"以组内出现频次最高的技能为代表生成「xxx技能包」建议标签，`original_skills` 列为合并前的原始技能（分号分隔）。"
    )
    if merge_path:
        intro_auto += f"\n源文件：`{_short_display_path(merge_path)}`。"
    else:
        intro_auto += "\n（当前未找到 skill_merge_preview.csv，请先运行 ETL 或执行项目根目录的 data_transfer_to_dashboard.py。）"

    cur_intro = (sections.get("appendix_skill_merge_intro") or "").strip()
    if not cur_intro:
        sections["appendix_skill_merge_intro"] = intro_auto

    return sections


def files_status() -> List[Dict[str, Any]]:
    """Per-artifact readiness: standard `data/` vs effective path actually used by loaders.

    Each item: keys
      - key: short id for code/tests
      - label: primary filename shown in UI (effective basename when found)
      - ok: whether any path resolves for this artifact
      - standard: list of {name, path, ok} for files checked only under data/
      - effective_path, effective_rel, source_bucket (when ok)
    """
    rows: List[Dict[str, Any]] = []

    # --- Jobs ---
    job_standard_names = ("jobs_clean.csv", "job_vec.csv")
    job_standard = []
    for n in job_standard_names:
        p = os.path.join(DATA_DIR, n)
        job_standard.append({"name": n, "path": p, "ok": os.path.isfile(p)})
    job_eff = _resolve_jobs_csv_path()
    rows.append(
        {
            "key": "jobs",
            "label": os.path.basename(job_eff) if job_eff else "job_vec.csv / jobs_clean.csv",
            "ok": bool(job_eff),
            "standard": job_standard,
            "effective_path": job_eff,
            "effective_rel": _short_display_path(job_eff),
            "source_bucket": _source_bucket(job_eff),
        }
    )

    # --- Cluster profiles (industry penetration) ---
    cp_name = "cluster_profiles.csv"
    cp_std = os.path.join(DATA_DIR, cp_name)
    cp_eff = _resolve_cluster_profiles_path()
    rows.append(
        {
            "key": "cluster_profiles",
            "label": cp_name,
            "ok": bool(cp_eff),
            "standard": [{"name": cp_name, "path": cp_std, "ok": os.path.isfile(cp_std)}],
            "effective_path": cp_eff,
            "effective_rel": _short_display_path(cp_eff),
            "source_bucket": _source_bucket(cp_eff),
        }
    )

    # --- Skill value (robust regression output) ---
    sk_name = "skill_value_robust.csv"
    sk_std = os.path.join(DATA_DIR, sk_name)
    sk_eff = _resolve_skill_value_robust_path()
    rows.append(
        {
            "key": "skill_value_robust",
            "label": sk_name,
            "ok": bool(sk_eff),
            "standard": [{"name": sk_name, "path": sk_std, "ok": os.path.isfile(sk_std)}],
            "effective_path": sk_eff,
            "effective_rel": _short_display_path(sk_eff),
            "source_bucket": _source_bucket(sk_eff),
        }
    )

    # --- Skill impact (MI / mixed-effects, large-n industries) ---
    si_name = "skill_impact.csv"
    si_std = os.path.join(DATA_DIR, si_name)
    si_eff = _resolve_skill_impact_path()
    rows.append(
        {
            "key": "skill_impact",
            "label": si_name,
            "ok": bool(si_eff),
            "standard": [{"name": si_name, "path": si_std, "ok": os.path.isfile(si_std)}],
            "effective_path": si_eff,
            "effective_rel": _short_display_path(si_eff),
            "source_bucket": _source_bucket(si_eff),
        }
    )

    # --- Experience curve ---
    ex_name = "exp_curve.csv"
    ex_std = os.path.join(DATA_DIR, ex_name)
    ex_eff = _resolve_exp_curve_csv_path()
    rows.append(
        {
            "key": "exp_curve",
            "label": ex_name,
            "ok": bool(ex_eff),
            "standard": [{"name": ex_name, "path": ex_std, "ok": os.path.isfile(ex_std)}],
            "effective_path": ex_eff,
            "effective_rel": _short_display_path(ex_eff),
            "source_bucket": _source_bucket(ex_eff),
        }
    )

    # --- Optional: adapted cluster_results / skill_importance in data/ ---
    cr_std = os.path.join(DATA_DIR, "cluster_results.csv")
    cr_eff = _find(
        [
            cr_std,
            os.path.join(PARENT_DIR, "cluster_results.csv"),
            os.path.join(PARENT_DIR, "clustered_output", "cluster_results.csv"),
        ]
    )
    rows.append(
        {
            "key": "cluster_results",
            "label": "cluster_results.csv（可选，适配层）",
            "ok": bool(cr_eff),
            "standard": [{"name": "cluster_results.csv", "path": cr_std, "ok": os.path.isfile(cr_std)}],
            "effective_path": cr_eff,
            "effective_rel": _short_display_path(cr_eff),
            "source_bucket": _source_bucket(cr_eff),
        }
    )

    si_std = os.path.join(DATA_DIR, "skill_importance.csv")
    si_eff = _find(
        [
            si_std,
            os.path.join(PARENT_DIR, "skill_importance.csv"),
            os.path.join(PARENT_DIR, "clustered_output", "skill_importance.csv"),
        ]
    )
    rows.append(
        {
            "key": "skill_importance",
            "label": "skill_importance.csv（可选，适配层）",
            "ok": bool(si_eff),
            "standard": [{"name": "skill_importance.csv", "path": si_std, "ok": os.path.isfile(si_std)}],
            "effective_path": si_eff,
            "effective_rel": _short_display_path(si_eff),
            "source_bucket": _source_bucket(si_eff),
        }
    )

    # --- Skill merge preview (ETL) ---
    sm_name = "skill_merge_preview.csv"
    sm_std = os.path.join(DATA_DIR, sm_name)
    sm_eff = _resolve_skill_merge_preview_path()
    rows.append(
        {
            "key": "skill_merge_preview",
            "label": sm_name,
            "ok": bool(sm_eff),
            "standard": [{"name": sm_name, "path": sm_std, "ok": os.path.isfile(sm_std)}],
            "effective_path": sm_eff,
            "effective_rel": _short_display_path(sm_eff),
            "source_bucket": _source_bucket(sm_eff),
        }
    )

    # --- Report JSON ---
    rep_std = os.path.join(DATA_DIR, "report_sections.json")
    rep_eff = _find([rep_std, os.path.join(PARENT_DIR, "report_sections.json")])
    rows.append(
        {
            "key": "report_sections",
            "label": "report_sections.json",
            "ok": bool(rep_eff),
            "standard": [{"name": "report_sections.json", "path": rep_std, "ok": os.path.isfile(rep_std)}],
            "effective_path": rep_eff,
            "effective_rel": _short_display_path(rep_eff),
            "source_bucket": _source_bucket(rep_eff),
        }
    )

    return rows


if __name__ == '__main__':
    print('data_bridge loaded. Data dir:', DATA_DIR)
