"""图表读图说明：基于与作图相同的数据做本地摘要，可选调用 DeepSeek API 增强。"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd


def _fmt_money(x: float) -> str:
    if pd.isna(x):
        return '—'
    return f"¥{float(x):,.0f}"


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode('utf-8', errors='replace'))
    return h.hexdigest()[:20]


def insight_industry_salary_bar(grp: pd.DataFrame, group_col: str) -> str:
    """行业 × 平均薪资 条形图（与 px.bar 使用同一 grp 子集）。"""
    if grp.empty or group_col not in grp.columns:
        return '暂无数据可解读。'
    g = grp.dropna(subset=['avg_salary']).copy()
    if g.empty:
        return '平均薪资均为缺失，无法比较行业间差异。'
    top = g.loc[g['avg_salary'].idxmax()]
    bot = g.loc[g['avg_salary'].idxmin()]
    med = float(g['avg_salary'].median())
    spread = float(g['avg_salary'].max() - g['avg_salary'].min())
    lines = [
        f'图中展示 **{len(g)}** 个行业（按岗位量取前若干名后排序作图），纵轴为样本内平均月薪（元/月）。',
        f'**薪资最高**行业为「{top[group_col]}」（约 {_fmt_money(top["avg_salary"])}）；**最低**为「{bot[group_col]}」（约 {_fmt_money(bot["avg_salary"])}）。',
        f'中位数约 **{_fmt_money(med)}**，极差约 **{_fmt_money(spread)}**，可粗略判断行业间薪资离散程度；条形越长表示该行业在样本中平均出价越高（仍受岗位结构、城市、职级混杂影响）。',
    ]
    return '\n\n'.join(lines)


def insight_industry_count_bar(grp: pd.DataFrame, group_col: str) -> str:
    """行业 × 岗位数量 条形图。"""
    if grp.empty or 'count' not in grp.columns:
        return '暂无数据可解读。'
    g = grp.sort_values('count', ascending=False)
    top = g.iloc[0]
    total = int(g['count'].sum())
    share = float(top['count'] / total * 100) if total else 0.0
    lines = [
        f'纵轴为各行业岗位条数（与上图同一批行业，便于对照「量」与「价」）。样本内岗位合计 **{total}** 条。',
        f'**岗位最多**行业为「{top[group_col]}」（{int(top["count"])} 条，约占 **{share:.1f}%**）。',
        '若某行业「数量高但薪资图里并不突出」，多为岗位供给大、薪资中位数被稀释；反之则可能为高薪小众赛道。',
    ]
    return '\n\n'.join(lines)


def insight_education_salary(summary: pd.DataFrame) -> str:
    """学历 × 平均薪资。"""
    if summary.empty:
        return '暂无数据可解读。'
    s = summary.dropna(subset=['avg_salary']).copy()
    if s.empty:
        return '各学历平均薪资缺失较多。'
    hi = s.iloc[0]
    lo = s.iloc[-1]
    nmax = int(s['count'].max()) if 'count' in s.columns else 0
    edu_max_n = s.loc[s['count'].idxmax(), 'education'] if 'count' in s.columns and s['count'].notna().any() else '—'
    lines = [
        '柱状按**平均薪资从高到低**排列，颜色越深表示均值越高（样本内）。',
        f'均薪最高档为 **{hi["education"]}**（约 {_fmt_money(hi["avg_salary"])}）；最低为 **{lo["education"]}**（约 {_fmt_money(lo["avg_salary"])}）。',
    ]
    if 'count' in s.columns and nmax > 0:
        lines.append(f'样本量最大的是 **{edu_max_n}**（{nmax} 条），解读时需结合「人数多寡」与极端值对均值的影响。')
    return '\n\n'.join(lines)


def _prefix_redundant_city_names(names: list[str]) -> set[str]:
    """若存在更长地名 m 以「n·」开头，则 n 视为被区县细化覆盖的短名，不再单独展示（如 上海 vs 上海·浦东新区）。"""
    sep = '\u00b7'  # 中文间隔号 ·
    uniq = list(dict.fromkeys(names))
    sset = set(uniq)
    redundant: set[str] = set()
    for n in uniq:
        for m in sset:
            if m == n:
                continue
            if m.startswith(n + sep):
                redundant.add(n)
                break
    return redundant


def _city_value_counts_top_filtered(plot_df: pd.DataFrame, pool: int = 30, final_k: int = 5) -> list[tuple[str, int]]:
    """按岗位数从高到低取地名，去掉「另一地名以其 + · 为前缀」的短名后再取前 final_k 条。"""
    if 'city' not in plot_df.columns or plot_df.empty:
        return []
    vc = plot_df['city'].astype(str).value_counts()
    pool_names = [str(x) for x in vc.head(pool).index.tolist()]
    redundant = _prefix_redundant_city_names(pool_names)
    picked: list[tuple[str, int]] = []
    for name, cnt in vc.items():
        sn = str(name)
        if sn in redundant:
            continue
        picked.append((sn, int(cnt)))
        if len(picked) >= final_k:
            break
    return picked


def insight_geo_points(plot_df: pd.DataFrame, geo_df: pd.DataFrame) -> str:
    """地理散点（每点一条岗位）。"""
    if plot_df.empty:
        return '无落点数据。'
    n = len(plot_df)
    if 'city' in plot_df.columns:
        top_list = _city_value_counts_top_filtered(plot_df, pool=30, final_k=5)
        topc = '、'.join([f'{k}（{int(v)}）' for k, v in top_list]) if top_list else '—'
        geo_line = f'落点最多的城市/区县（city 字段 Top5，已省略被「·」下级地名覆盖的短名）：{topc}。'
    else:
        geo_line = ''
    lines = [
        f'当前图共 **{n}** 个散点，对应 **{n}** 条可定位岗位（占用于作图的记录 **{n}/{len(geo_df)}**）。',
        '同一城市内多条岗位会带微量随机偏移，避免点完全重叠；悬停在点上可看单条岗位信息。',
        geo_line,
    ]
    return '\n\n'.join([x for x in lines if x])


def insight_skill_importance(skill_imp: pd.DataFrame, coarse: bool = False) -> str:
    """技能条形（importance 已归一化或溢价代理）。"""
    if skill_imp.empty:
        return '暂无技能维度数据。'
    s = skill_imp.sort_values('importance', ascending=False).head(8)
    tops = '、'.join(s['skill'].astype(str).tolist())
    lines = [
        '（大样本）纵轴为模型给出的相对重要性或稳健溢价代理，已按条形排序，仅作**横向比较**用。',
        f'排名前若干的技能包括：**{tops}** 等。',
    ]
    if coarse:
        lines.append('（中小样本）以下为粗匹配/中位数差异类指标，解释力弱于大样本回归，建议谨慎外推。')
    return '\n\n'.join(lines)


def insight_exp_salary_curve(exp: pd.DataFrame) -> str:
    """经验 × 预测薪资折线。"""
    if exp.empty or 'salary' not in exp.columns:
        return '暂无曲线数据。'
    e = exp.copy()
    if 'exp_years' not in e.columns:
        if 'years_experience' in e.columns:
            e['exp_years'] = pd.to_numeric(e['years_experience'], errors='coerce')
        elif 'year' in e.columns:
            e['exp_years'] = pd.to_numeric(e['year'], errors='coerce') - 2023
        else:
            e['exp_years'] = pd.Series(range(len(e)), dtype=float)
    e = e.dropna(subset=['exp_years', 'salary']).sort_values('exp_years')
    if e.empty:
        return '暂无曲线数据。'
    s0 = float(e['salary'].iloc[0])
    s1 = float(e['salary'].iloc[-1])
    x0 = float(e['exp_years'].iloc[0])
    x1 = float(e['exp_years'].iloc[-1])
    growth = (s1 / s0 - 1.0) * 100 if s0 else 0.0
    lines = [
        '横轴为经验年数，纵轴为模型预测月薪（与回归输出一致）。',
        f'曲线从 **{x0:.0f} 年经验**（约 {_fmt_money(s0)}）到 **{x1:.0f} 年经验**（约 {_fmt_money(s1)}），相对变化约 **{growth:+.1f}%**（为样本内模型外推，非个体因果）。',
    ]
    return '\n\n'.join(lines)


def deepseek_chart_note(
    chart_title: str,
    local_summary: str,
    data_csv_snippet: str,
    api_key: str,
    base_url: Optional[str] = None,
    model: str = 'deepseek-chat',
    timeout: int = 60,
) -> str:
    """调用 DeepSeek Chat Completions（OpenAI 兼容）。失败时返回空串。"""
    api_key = (api_key or '').strip()
    if not api_key:
        return ''
    base = (base_url or os.environ.get('DEEPSEEK_API_BASE', 'https://api.deepseek.com')).rstrip('/')
    url = f'{base}/v1/chat/completions'
    system = (
        '你是数据分析助手，根据用户给出的图表标题、读图要点摘要和一小段作图用数据（CSV 片段），'
        '用 2～4 句中文补充「读图要点」：趋势、对比、可能偏差或使用建议；不要编造数据中不存在的具体数字；'
        '若数据过少请直接说明无法深入。不要 Markdown 一级标题。'
    )
    user = f'【图表】{chart_title}\n\n【已有要点】\n{local_summary}\n\n【数据片段】\n{data_csv_snippet}'
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'max_tokens': 400,
        'temperature': 0.4,
    }
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode('utf-8'))
        return (raw.get('choices') or [{}])[0].get('message', {}).get('content', '').strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError, KeyError, IndexError):
        return ''


def df_snippet(df: pd.DataFrame, max_rows: int = 12, max_chars: int = 3500) -> str:
    """给 LLM 的短 CSV 文本（控制长度）。"""
    if df is None or df.empty:
        return '(空表)'
    s = df.head(max_rows).to_csv(index=False)
    if len(s) > max_chars:
        s = s[:max_chars] + '\n...(截断)'
    return s


def digest_for_df(df: pd.DataFrame, *extra: str) -> str:
    if df is None or df.empty:
        return _digest('empty', *extra)
    return _digest(df_snippet(df, max_rows=50, max_chars=8000), *extra)
