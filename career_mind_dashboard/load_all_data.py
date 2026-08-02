import os
from career_mind_dashboard import data_bridge as db
import pandas as pd

print('files status:')
for row in db.files_status():
    print(f" - {row.get('label')}: ok={row.get('ok')}")
    for s in row.get('standard') or []:
        print(f"     data/{s.get('name')}: {s.get('ok')}")
    if row.get('ok'):
        print(f"     effective: {row.get('effective_rel')} ({row.get('source_bucket')})")

print('\nLoading jobs...')
jobs = db.load_jobs()
print(' jobs rows:', len(jobs))
print(' jobs columns:', jobs.columns.tolist())

print('\nChecking cluster_results...')
try:
    import_path = os.path.join(os.path.dirname(__file__), 'data', 'cluster_results.csv')
    if os.path.exists(import_path):
        cr = pd.read_csv(import_path)
        print(' cluster_results rows:', len(cr))
        print(' cluster_results cols:', cr.columns.tolist())
    else:
        print(' cluster_results.csv not found in data/')
except Exception as e:
    print(' error reading cluster_results:', e)

print('\nChecking skill_importance...')
try:
    spath = os.path.join(os.path.dirname(__file__), 'data', 'skill_importance.csv')
    if os.path.exists(spath):
        sk = pd.read_csv(spath)
        print(' skill_importance rows:', len(sk))
        print(' skill_importance cols:', sk.columns.tolist())
        # sample top skills per cluster
        if 'cluster_uid' in sk.columns:
            print('\n sample cluster_uid values:', sk['cluster_uid'].unique()[:5])
    else:
        print(' skill_importance.csv not found in data/')
except Exception as e:
    print(' error reading skill_importance:', e)

print('\nChecking exp_curve...')
try:
    epath = os.path.join(os.path.dirname(__file__), 'data', 'exp_curve.csv')
    if os.path.exists(epath):
        ex = pd.read_csv(epath)
        print(' exp_curve rows:', len(ex))
        print(' exp_curve cols:', ex.columns.tolist())
        if 'salary' in ex.columns:
            vals = ex['salary'].describe()
            print(' salary summary:\n', vals)
    else:
        print(' exp_curve.csv not found in data/')
except Exception as e:
    print(' error reading exp_curve:', e)

print('\nReport sections:')
rs = db.load_report_sections()
for k,v in rs.items():
    print(f' - {k}:', str(v)[:140])

# Basic validation suggestions
print('\nValidation checks:')
if 'salary_avg' in jobs.columns:
    if jobs['salary_avg'].dtype.kind not in 'biufc':
        print(' - salary_avg is not numeric; consider converting or check ETL')
else:
    print(' - salary_avg missing in jobs')

if os.path.exists(import_path):
    cr = pd.read_csv(import_path)
    if 'salary_min_avg' in cr.columns and cr['salary_min_avg'].max() < 1000:
        print(' - NOTE: cluster_results.salary_min_avg shows small values (maybe in k); ETL likely handles scaling.')

print('\nLoad complete')
