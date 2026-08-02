#!/usr/bin/env python3
"""Adapt clustered_output CSVs into career_mind_dashboard/data/ for the dashboard.

This script will look for files under the repository's `clustered_output/`
folder and generate adapted CSVs under `career_mind_dashboard/data/`:

- cluster_results.csv  (cluster_uid, cluster_id, industry, core_skills, salary_min_avg, salary_max_avg, count, sample_size)
- skill_importance.csv (cluster_uid, cluster_id, industry, skill, importance)
- exp_curve.csv        (cluster_uid, year, salary)

Run this from the repository root (or from the analysis folder):
  python career_mind_dashboard/adapt_clustered_output.py

The script is resilient: if some source files are missing it will skip them
and still produce whatever adapted outputs it can.
"""

import os
import sys
import math
from pathlib import Path

import pandas as pd
import numpy as np


HERE = Path(__file__).parent.resolve()
REPO_ROOT = HERE.parent
CLUSTERED_DIR = REPO_ROOT / 'clustered_output'
DATA_DIR = HERE / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_profiles():
    for fname in ('cluster_profiles.csv', 'cluster_profiles-utf-8.csv'):
        path = CLUSTERED_DIR / fname
        if path.exists():
            print('Reading', path)
            return pd.read_csv(path, low_memory=False)
    print('No cluster_profiles file found in', CLUSTERED_DIR)
    return None


def adapt_cluster_results(df_profiles: pd.DataFrame):
    df = df_profiles.copy()
    if 'industry_group' in df.columns and 'industry' not in df.columns:
        df = df.rename(columns={'industry_group': 'industry'})
    if 'job_count' in df.columns:
        df['count'] = df['job_count']
    if 'count' not in df.columns:
        df['count'] = df.get('sample_size', 0)
    df['sample_size'] = df['count']
    df['cluster_uid'] = df['industry'].astype(str) + '__' + df['cluster_id'].astype(str)
    out_cols = ['cluster_uid', 'cluster_id', 'industry', 'core_skills', 'salary_min_avg', 'salary_max_avg', 'count', 'sample_size']
    out = df.reindex(columns=out_cols)
    out_path = DATA_DIR / 'cluster_results.csv'
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print('Wrote', out_path)
    return out


def adapt_skill_importance(df_profiles: pd.DataFrame):
    skill_path = CLUSTERED_DIR / 'skill_impact.csv'
    if not skill_path.exists():
        print('skill_impact.csv not found, skipping skill_importance adaptation')
        return None
    print('Reading', skill_path)
    df_skill = pd.read_csv(skill_path, low_memory=False)
    if 'industry_group' in df_skill.columns and 'industry' not in df_skill.columns:
        df_skill = df_skill.rename(columns={'industry_group': 'industry'})
    # ensure numeric coefficient
    if 'coefficient' in df_skill.columns:
        df_skill['coefficient'] = pd.to_numeric(df_skill['coefficient'], errors='coerce')
    else:
        df_skill['coefficient'] = pd.to_numeric(df_skill.get('coef', pd.Series([np.nan]*len(df_skill))), errors='coerce')

    df_skill = df_skill.dropna(subset=['coefficient', 'skill', 'industry'])
    rows = []
    for industry, grp in df_skill.groupby('industry'):
        clusters = df_profiles[df_profiles['industry'] == industry]
        if clusters.empty:
            # if no matching industry in profiles, still include industry-level rows with cluster_uid=industry__0
            cluster_uid = f"{industry}__0"
            for _, srow in grp.iterrows():
                rows.append({'cluster_uid': cluster_uid, 'cluster_id': 0, 'industry': industry, 'skill': srow['skill'], 'importance': abs(float(srow['coefficient']))})
            continue
        for _, prow in clusters.iterrows():
            cluster_uid = f"{prow['industry']}__{prow['cluster_id']}"
            for _, srow in grp.iterrows():
                rows.append({'cluster_uid': cluster_uid, 'cluster_id': prow['cluster_id'], 'industry': industry, 'skill': srow['skill'], 'importance': abs(float(srow['coefficient']))})

    if not rows:
        print('No skill rows generated')
        return None

    df_out = pd.DataFrame(rows)
    # normalize importance per cluster_uid
    df_out['importance'] = df_out.groupby('cluster_uid')['importance'].transform(lambda x: x / x.sum() if x.sum() > 0 else x)
    out_path = DATA_DIR / 'skill_importance.csv'
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print('Wrote', out_path)
    return df_out


def adapt_exp_curve(df_profiles: pd.DataFrame):
    exp_path = CLUSTERED_DIR / 'exp_curve.csv'
    if not exp_path.exists():
        print('exp_curve.csv not found, skipping exp_curve adaptation')
        return None
    print('Reading', exp_path)
    df_exp = pd.read_csv(exp_path, low_memory=False)
    if 'industry_group' in df_exp.columns and 'industry' not in df_exp.columns:
        df_exp = df_exp.rename(columns={'industry_group': 'industry'})

    rows = []
    for industry, grp in df_exp.groupby('industry'):
        clusters = df_profiles[df_profiles['industry'] == industry]
        if clusters.empty:
            cluster_ids = [0]
            cluster_uids = [f"{industry}__0"]
        else:
            cluster_ids = clusters['cluster_id'].tolist()
            cluster_uids = (clusters['industry'].astype(str) + '__' + clusters['cluster_id'].astype(str)).tolist()

        for cluster_id, uid in zip(cluster_ids, cluster_uids):
            for _, erow in grp.iterrows():
                years = erow.get('years_experience') if 'years_experience' in erow else erow.get('year', None)
                if pd.isna(years):
                    # some files use 'year' already
                    years = erow.get('year', None)
                if years is None or pd.isna(years):
                    continue
                year = int(2023 + float(years))
                pred = erow.get('predicted_salary') if 'predicted_salary' in erow else erow.get('salary', None)
                if pred is None or pd.isna(pred):
                    continue
                salary = float(pred)
                # scale small predicted values to RMB if clearly fractional
                if abs(salary) < 1000:
                    salary = salary * 1000.0
                rows.append({'cluster_uid': uid, 'cluster_id': cluster_id, 'year': year, 'salary': salary})

    if not rows:
        print('No exp rows generated')
        return None

    df_out = pd.DataFrame(rows)
    out_path = DATA_DIR / 'exp_curve.csv'
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print('Wrote', out_path)
    return df_out


def main():
    df_profiles = read_profiles()
    if df_profiles is None:
        print('No profiles present; aborting adaptation.')
        return
    df_profiles_std = df_profiles.copy()
    if 'industry_group' in df_profiles_std.columns and 'industry' not in df_profiles_std.columns:
        df_profiles_std = df_profiles_std.rename(columns={'industry_group': 'industry'})
    if 'job_count' in df_profiles_std.columns:
        df_profiles_std['count'] = df_profiles_std['job_count']
    df_profiles_std['sample_size'] = df_profiles_std.get('count', df_profiles_std.get('sample_size', 0))

    adapt_cluster_results(df_profiles_std)
    adapt_skill_importance(df_profiles_std)
    adapt_exp_curve(df_profiles_std)


if __name__ == '__main__':
    main()
