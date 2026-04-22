import os
import re
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_regression
import statsmodels.formula.api as smf

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INPUT_CSV = os.path.join(DATA_DIR, "job_vec.csv")
OUTPUT_FOLDER = os.path.join(DATA_DIR, "regression_output")
TOP_N_SKILLS_MAX = 16
TOP_N_SKILLS_MIN = 5
SVD_COMPONENTS = 3
REGRESSION_MIN_SAMPLES = 70
ROBUST_MIN_SAMPLES = 10
MUTUAL_INFO_RANDOM_STATE = 42
EXP_CURVE_POWER = 1.5  # use a slightly sub-quadratic experience term for smoother salary growth


def extract_skills(skill_str):
    """Split skills text into a list of tokens."""
    if pd.isna(skill_str):
        return []
    return [token.strip() for token in re.split(r'[,;；]+', str(skill_str)) if token.strip()]


def safe_skill_name(skill):
    """Convert raw skill phrase into a safe column name."""
    safe = re.sub(r'[^0-9a-zA-Z]+', '_', skill)
    safe = re.sub(r'_+', '_', safe).strip('_')
    return f'skill_{safe}' if safe else f'skill_empty'


def build_skill_matrix(df_group):
    """Build binary skill matrix for skills that appear at least twice."""
    all_skills = []
    for skills_list in df_group['skills_list']:
        all_skills.extend(skills_list)

    skill_counts = Counter(all_skills)
    valid_skills = [skill for skill, count in skill_counts.items() if count >= 2]
    if not valid_skills:
        return pd.DataFrame(index=df_group.index), [], {}

    safe_names = {}
    used_names = set()
    for skill in valid_skills:
        base_name = safe_skill_name(skill)
        name = base_name
        suffix = 1
        while name in used_names:
            name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(name)
        safe_names[skill] = name

    matrix = pd.DataFrame(0, index=df_group.index, columns=list(safe_names.values()))
    for idx, skills_list in enumerate(df_group['skills_list']):
        for skill in skills_list:
            if skill in safe_names:
                matrix.loc[df_group.index[idx], safe_names[skill]] = 1

    return matrix, valid_skills, safe_names


def select_mutual_info_skills(df_group, skill_matrix, valid_skills, safe_names, top_n_skills):
    """Select top skills by mutual information with average salary."""
    if skill_matrix.shape[1] == 0:
        return [], []

    y = df_group['avg_salary'].values
    mi_scores = mutual_info_regression(skill_matrix, y, discrete_features=True, random_state=MUTUAL_INFO_RANDOM_STATE)
    skill_scores = []
    for skill in valid_skills:
        safe_col = safe_names[skill]
        idx = list(skill_matrix.columns).index(safe_col)
        skill_scores.append((safe_col, skill, mi_scores[idx]))

    skill_scores.sort(key=lambda x: x[2], reverse=True)
    selected_columns = [col for col, skill, _ in skill_scores[:top_n_skills]]
    selected_skills = [skill for col, skill, _ in skill_scores[:top_n_skills]]
    return selected_columns, selected_skills


def build_group_skill_budget(df):
    """Create per-industry skill-count budget by sample-size rank (16 -> 5)."""
    group_sizes = df.groupby('industry_group').size().sort_values(ascending=False)
    eligible_groups = group_sizes[group_sizes >= REGRESSION_MIN_SAMPLES]
    if eligible_groups.empty:
        return {}

    budgets = np.linspace(TOP_N_SKILLS_MAX, TOP_N_SKILLS_MIN, num=len(eligible_groups))
    budget_map = {}
    for group_name, budget in zip(eligible_groups.index, budgets):
        budget_map[group_name] = int(np.clip(np.rint(budget), TOP_N_SKILLS_MIN, TOP_N_SKILLS_MAX))
    return budget_map


def build_residual_features(df_group, selected_skills):
    """Create residual context using leftover skills and SVD."""
    residual_texts = []
    for skills_list in df_group['skills_list']:
        leftover = [skill for skill in skills_list if skill not in selected_skills]
        residual_texts.append(' '.join(leftover) if leftover else '')

    if all(text == '' for text in residual_texts):
        return pd.DataFrame(0, index=df_group.index, columns=[f'residual_context_{i+1}' for i in range(SVD_COMPONENTS)])

    try:
        vectorizer = TfidfVectorizer(token_pattern=r'[^ ]+', lowercase=False, max_features=200)
        tfidf_matrix = vectorizer.fit_transform(residual_texts)
        n_comp = min(SVD_COMPONENTS, tfidf_matrix.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        residual_features = svd.fit_transform(tfidf_matrix)
        if residual_features.shape[1] < SVD_COMPONENTS:
            padding = np.zeros((residual_features.shape[0], SVD_COMPONENTS - residual_features.shape[1]))
            residual_features = np.hstack([residual_features, padding])
        return pd.DataFrame(residual_features, index=df_group.index, columns=[f'residual_context_{i+1}' for i in range(SVD_COMPONENTS)])
    except Exception:
        return pd.DataFrame(0, index=df_group.index, columns=[f'residual_context_{i+1}' for i in range(SVD_COMPONENTS)])


def fit_mixed_effects(df_group, skill_matrix, selected_columns, selected_skills, group_name):
    """Fit MixedLM with random slopes for experience and fallback to OLS."""
    model_data = df_group[['avg_salary', 'exp_years']].copy()
    model_data['exp_smooth'] = np.power(model_data['exp_years'], EXP_CURVE_POWER)
    skill_data = skill_matrix[selected_columns].copy()
    residual_data = build_residual_features(df_group, selected_skills)
    model_data = pd.concat([model_data, skill_data, residual_data], axis=1)

    fixed_cols = ['exp_years', 'exp_smooth'] + list(skill_data.columns) + list(residual_data.columns)
    formula = 'avg_salary ~ ' + ' + '.join(fixed_cols)

    try:
        model = smf.mixedlm(formula, model_data, groups=df_group['industry_group'], re_formula='~exp_years + exp_smooth')
        result = model.fit(method='lbfgs', reml=True)
        return result, model_data, fixed_cols
    except Exception as e:
        print(f"  MixedLM failed for {group_name}, falling back to OLS: {e}")
        try:
            ols_model = smf.ols(formula, model_data)
            ols_result = ols_model.fit()
            return ols_result, model_data, fixed_cols
        except Exception as e2:
            print(f"  OLS fallback failed for {group_name}: {e2}")
            return None, None, None


def assign_exp_bucket(exp_years):
    """Map experience years to binary buckets using 4-year threshold."""
    if pd.isna(exp_years):
        return np.nan
    if exp_years < 4:
        return '<4'
    return '>=4'


def compute_robust_skill_values(df_group, skill_matrix, valid_skills, safe_names):
    """For medium samples, estimate pure skill value by median premium + coarse matching."""
    robust_rows = []
    work_df = df_group[['avg_salary', 'exp_years']].copy()
    work_df['exp_bucket'] = work_df['exp_years'].apply(assign_exp_bucket)

    for raw_skill in valid_skills:
        col_name = safe_names[raw_skill]
        if col_name not in skill_matrix.columns:
            continue

        skill_flag = skill_matrix[col_name]
        n_with_skill = int(skill_flag.sum())
        n_without_skill = int((1 - skill_flag).sum())
        if n_with_skill < 2 or n_without_skill < 2:
            continue

        with_skill_salary = work_df.loc[skill_flag == 1, 'avg_salary']
        without_skill_salary = work_df.loc[skill_flag == 0, 'avg_salary']
        median_premium = with_skill_salary.median() - without_skill_salary.median()

        bucket_diffs = []
        bucket_weights = []
        for bucket_name in ['<4', '>=4']:
            bucket_mask = work_df['exp_bucket'] == bucket_name
            if bucket_mask.sum() == 0:
                continue

            in_bucket_with = work_df.loc[bucket_mask & (skill_flag == 1), 'avg_salary']
            in_bucket_without = work_df.loc[bucket_mask & (skill_flag == 0), 'avg_salary']
            if len(in_bucket_with) < 1 or len(in_bucket_without) < 1:
                continue

            bucket_diff = in_bucket_with.median() - in_bucket_without.median()
            bucket_diffs.append(bucket_diff)
            bucket_weights.append(float(len(in_bucket_with)))

        coarse_match_premium = np.nan
        if bucket_diffs:
            coarse_match_premium = float(np.average(bucket_diffs, weights=bucket_weights))

        pure_skill_value = coarse_match_premium if pd.notna(coarse_match_premium) else float(median_premium)
        if pd.notna(coarse_match_premium) and np.sign(median_premium) != np.sign(coarse_match_premium):
            continue

        robust_rows.append({
            'industry_group': df_group['industry_group'].iloc[0],
            'skill': raw_skill,
            'n_samples': len(df_group),
            'n_with_skill': n_with_skill,
            'n_without_skill': n_without_skill,
            'median_premium': float(median_premium),
            'coarse_match_premium': coarse_match_premium,
            'pure_skill_value': float(pure_skill_value),
            'matched_bucket_count': len(bucket_diffs),
            'method': 'median_plus_coarse_matching'
        })

    return robust_rows


def main():

    df = pd.read_csv(INPUT_CSV)
    # ETL 已统一薪资单位与经验数字化；这里直接复用预处理结果
    if 'job_vec_0' not in df.columns:
        raise ValueError("Missing required column 'job_vec_0' in job_vec.csv")

    # 只保留经验标量向量列，跳过其余 job_vec_n (n>0) 向量列
    drop_vec_cols = [c for c in df.columns if c.startswith('job_vec_') and c != 'job_vec_0']
    if drop_vec_cols:
        df = df.drop(columns=drop_vec_cols)

    if 'avg_salary' not in df.columns:
        if {'salary_min', 'salary_max'}.issubset(df.columns):
            df['avg_salary'] = (df['salary_min'] + df['salary_max']) / 2
        elif {'salary_min_norm', 'salary_max_norm'}.issubset(df.columns):
            df['avg_salary'] = (df['salary_min_norm'] + df['salary_max_norm']) / 2
        else:
            raise ValueError("Missing salary columns: expected 'avg_salary' or min/max salary columns.")

    # job_vec_0 是“以十年为单位”的小数，这里换算回“年”
    df['exp_years'] = df['job_vec_0'] * 10.0
    skill_source_col = 'merged_job_skills' if 'merged_job_skills' in df.columns else 'job_skills'
    df['skills_list'] = df[skill_source_col].apply(extract_skills)
    df = df.dropna(subset=['avg_salary', 'exp_years', 'industry_group'])
    df = df[df['avg_salary'] > 0]

    skill_impacts = []
    exp_curves = []
    robust_skill_values = []

    group_skill_budget = build_group_skill_budget(df)
    if not group_skill_budget:
        print(f"No industry group with n>={REGRESSION_MIN_SAMPLES}; regression stage skipped.")

    for group_name, df_group in df.groupby('industry_group'):
        group_size = len(df_group)

        if group_size >= ROBUST_MIN_SAMPLES and group_size < REGRESSION_MIN_SAMPLES:
            print(f"Processing {group_name} (n={group_size}, robust stats mode)...")
            skill_matrix, valid_skills, safe_names = build_skill_matrix(df_group)
            if skill_matrix.shape[1] == 0:
                print(f"  No candidate skills with count>=2 in {group_name}")
                continue
            robust_rows = compute_robust_skill_values(df_group, skill_matrix, valid_skills, safe_names)
            if len(robust_rows) <= 2:
                print(f"  Skipping {group_name}: only {len(robust_rows)} robust skills extracted")
                continue
            robust_skill_values.extend(robust_rows)
            continue

        if group_size < REGRESSION_MIN_SAMPLES:
            continue

        top_n_skills = group_skill_budget.get(group_name)
        if top_n_skills is None:
            continue

        print(f"Processing {group_name} (n={group_size}, top_n_skills={top_n_skills})...")
        skill_matrix, valid_skills, safe_names = build_skill_matrix(df_group)
        if skill_matrix.shape[1] == 0:
            print(f"  No candidate skills with count>=2 in {group_name}")
            continue

        selected_columns, selected_skills = select_mutual_info_skills(
            df_group, skill_matrix, valid_skills, safe_names, top_n_skills
        )
        if not selected_columns:
            print(f"  No skills selected by mutual information for {group_name}")
            continue

        result, model_data, fixed_cols = fit_mixed_effects(df_group, skill_matrix, selected_columns, selected_skills, group_name)
        if result is None:
            continue

        summary = result.summary()
        for raw_skill, col_name in zip(selected_skills, selected_columns):
            if col_name not in result.params.index:
                continue
            coef = result.params[col_name]
            se = result.bse.get(col_name, np.nan)
            pval = result.pvalues.get(col_name, np.nan)
            ci_lower = coef - 1.96 * se if pd.notna(se) else np.nan
            ci_upper = coef + 1.96 * se if pd.notna(se) else np.nan
            skill_impacts.append({
                'industry_group': group_name,
                'skill': raw_skill,
                'selection_method': 'Mutual Information',
                'coefficient': coef,
                'std_err': se,
                'p_value': pval,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_samples': len(df_group)
            })

        exp_range = np.linspace(0, 10, 11)
        for exp_val in exp_range:
            pred_row = {col: 0 for col in fixed_cols}
            pred_row['exp_years'] = exp_val
            pred_row['exp_smooth'] = exp_val ** EXP_CURVE_POWER
            pred_df = pd.DataFrame([pred_row])

            predicted_salary = float(result.predict(pred_df).iloc[0])
            exp_curves.append({
                'industry_group': group_name,
                'years_experience': exp_val,
                'predicted_salary': predicted_salary
            })

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if skill_impacts:
        pd.DataFrame(skill_impacts).to_csv(os.path.join(OUTPUT_FOLDER, 'skill_impact.csv'), index=False)
    if exp_curves:
        pd.DataFrame(exp_curves).to_csv(os.path.join(OUTPUT_FOLDER, 'exp_curve.csv'), index=False)
    if robust_skill_values:
        pd.DataFrame(robust_skill_values).to_csv(os.path.join(OUTPUT_FOLDER, 'skill_value_robust.csv'), index=False)


if __name__ == '__main__':
    main()