import career_mind_dashboard.data_bridge as db

_status_by_key = {r['key']: r for r in db.files_status()}
print('cluster_results present:', _status_by_key.get('cluster_results', {}).get('ok'))

# Try an industry that exists in the adapted data
industry = '计算机技术'
clusters = db.get_clusters_for_industry(industry)
print('\nClusters sample (industry code length):', len(industry))
if clusters is None:
    print('  No cluster results loaded')
elif clusters.empty:
    print('  Cluster dataframe is empty')
else:
    cols = [c for c in ('cluster_id', 'sample_size') if c in clusters.columns]
    print('cluster head:', clusters[cols].head().to_string(index=False))
    cid = clusters.iloc[0]['cluster_id']
    print('\nSample cluster_id:', cid)
    skills = db.get_skill_importance(cid, industry=industry)
    print('Skill importance rows:', 0 if skills is None else len(skills))
    exp = db.get_exp_curve(None, industry=industry)
    print('Exp curve rows:', 0 if exp is None else len(exp))
