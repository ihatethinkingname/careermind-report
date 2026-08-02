#!/usr/bin/env python3
import os
from datetime import datetime
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
TEMPLATE_FILE = HERE / 'report_template.html'
OUTPUT_PDF = HERE / 'CareerMind_Report.pdf'
SKILL_MERGE_CORRELATION_THRESHOLD_DOC = 0.75
# 与本报告正文、图表、回归与聚类分析一致的岗位样本量（清洗后用于建模与可视化的记录数）
REPORT_ANALYSIS_RECORDS = 1473


def load_skill_merge_preview_rows():
    candidates = [
        HERE / 'data' / 'skill_merge_preview.csv',
        REPO_ROOT / 'skill_merge_preview.csv',
    ]
    for p in candidates:
        if p.is_file():
            try:
                df = pd.read_csv(p, low_memory=False).fillna('')
                return df.to_dict('records')
            except Exception:
                return []
    return []


def load_report_figures(analysis_geography: str, analysis_education: str, analysis_industry: str):
    """按“图 + 对应段落”方式组织图表内容。"""
    industry_overview_src = Path(
        r"C:\Users\21481\.cursor\projects\c-Users-21481-my-world-projects-CareerMind-analysis\assets\c__Users_21481_AppData_Roaming_Cursor_User_workspaceStorage_5802d2a58ffe594a89ea4aa19c5505db_images_image-4eee6dfc-b60b-4124-bed9-7a03d45bdbf5.png"
    )
    cluster_example_src = Path(
        r"C:\Users\21481\.cursor\projects\c-Users-21481-my-world-projects-CareerMind-analysis\assets\c__Users_21481_AppData_Roaming_Cursor_User_workspaceStorage_5802d2a58ffe594a89ea4aa19c5505db_images_image-bfec0d2d-a0d5-4fea-adf3-2b93488d46dd.png"
    )
    cluster_skill_analysis = build_cluster_cross_industry_analysis()
    given_paths = [
        Path(r"C:\Users\21481\.cursor\projects\c-Users-21481-my-world-projects-CareerMind-analysis\assets\c__Users_21481_AppData_Roaming_Cursor_User_workspaceStorage_5802d2a58ffe594a89ea4aa19c5505db_images_image-856ea137-8aab-4dab-927d-942a398b9ace.png"),
        Path(r"C:\Users\21481\.cursor\projects\c-Users-21481-my-world-projects-CareerMind-analysis\assets\c__Users_21481_AppData_Roaming_Cursor_User_workspaceStorage_5802d2a58ffe594a89ea4aa19c5505db_images_image-009c5673-5f12-4efe-9351-c9ddddf1bea4.png"),
    ]

    figure_meta = [
        {
            'title': '图1 岗位地理分布',
            'meaning': (
                f'地图数据源：与本报告分析样本一致的有效岗位共 {REPORT_ANALYSIS_RECORDS} 条（构图时取全部已载入岗位，'
                '不受侧边栏筛选；可落点数取决于经纬度或城市映射是否完整；每条一个点，同城随机偏移约 ±0.20°纬 / ±0.20°经，此参数可调）。'
            ),
            'related_paragraph': analysis_geography,
        },
        {
            'title': '图2 学历与薪资关系',
            'meaning': '数据源覆盖当前报告样本中的全部可用岗位，按学历层级汇总平均薪资并展示层级差异。图中可用于识别不同学历对应的薪资中枢、离散程度与样本分布特征，为“学历门槛—薪酬回报”关系提供直观证据。',
            'related_paragraph': analysis_education,
        },
    ]
    figures = []
    for idx, p in enumerate(given_paths):
        if not p.is_file():
            continue
        meta = figure_meta[idx] if idx < len(figure_meta) else {
            'title': f'图{idx + 1} 网页分析图',
            'meaning': '展示网页端分析产出的可视化结果。',
            'related_paragraph': '',
        }
        figures.append(
            {
                'title': meta['title'],
                'meaning': meta['meaning'],
                'related_paragraph': meta['related_paragraph'],
                'src': p.resolve().as_uri(),
            }
        )
    exp_fig = build_industry_salary_experience_figure()
    if exp_fig:
        figures.append(
            {
                'title': '图4 各行业经验-薪资关系（0-10年）',
                'meaning': '基于分行业回归结果绘制 0-10 年经验区间的预测薪资曲线，重点比较不同行业的薪资增长斜率、起薪差异与成长拐点。该图用于评估行业成长性，而非仅比较静态起薪水平。',
                'related_paragraph': analysis_industry,
                'src': exp_fig,
            }
        )
    return (
        figures,
        (industry_overview_src.resolve().as_uri() if industry_overview_src.is_file() else ''),
        (cluster_example_src.resolve().as_uri() if cluster_example_src.is_file() else ''),
        cluster_skill_analysis,
    )


def _split_skills(raw: str):
    if not isinstance(raw, str):
        return []
    parts = [p.strip() for p in re.split(r'[;；]', raw) if p and p.strip()]
    cleaned = []
    for p in parts:
        p = p.replace('技能包', '').strip()
        if p and p not in cleaned:
            cleaned.append(p)
    return cleaned


def _parse_experience_years(exp_text: str) -> float:
    if not isinstance(exp_text, str) or not exp_text.strip():
        return float('nan')
    t = exp_text.strip()
    if '无需经验' in t:
        return 0.0
    nums = re.findall(r'\d+(?:\.\d+)?', t)
    if not nums:
        return float('nan')
    vals = [float(x) for x in nums]
    if len(vals) >= 2 and '-' in t:
        return sum(vals[:2]) / 2.0
    return vals[0]


def build_cluster_cross_industry_analysis():
    cluster_path = HERE / 'data' / 'cluster_profiles.csv'
    impact_path = HERE / 'data' / 'skill_impact.csv'
    robust_path = HERE / 'data' / 'skill_value_robust.csv'
    if not cluster_path.is_file():
        return ''
    try:
        df = pd.read_csv(cluster_path, low_memory=False)
    except Exception:
        return ''

    req_cols = {
        'industry_group',
        'cluster_id',
        'profile_name',
        'job_count',
        'core_skills',
        'experience',
        'salary_min_avg',
        'salary_max_avg',
    }
    if not req_cols.issubset(df.columns):
        return ''

    d = df.copy()
    d['job_count'] = pd.to_numeric(d['job_count'], errors='coerce')
    d['salary_min_avg'] = pd.to_numeric(d['salary_min_avg'], errors='coerce')
    d['salary_max_avg'] = pd.to_numeric(d['salary_max_avg'], errors='coerce')
    d = d.dropna(subset=['industry_group', 'cluster_id', 'job_count', 'salary_min_avg', 'salary_max_avg'])
    if d.empty:
        return ''

    d['salary_mid'] = (d['salary_min_avg'] + d['salary_max_avg']) / 2.0
    d['exp_years'] = d['experience'].astype(str).map(_parse_experience_years)

    weak_tokens = {
        '办公软件', '沟通能力', '管理', '销售', '数据分析', '学习能力', '责任心', '团队协作',
        '执行能力', '抗压能力', '办公', 'excel'
    }
    mgmt_tokens = {'管理', '经理', '总监', '主管', '运营'}

    def weak_skill_score(skills: str) -> float:
        ks = _split_skills(skills)
        if not ks:
            return 0.0
        hit = sum(1 for k in ks if any(tok in k for tok in weak_tokens))
        return hit / len(ks)

    d['weak_skill_score'] = d['core_skills'].astype(str).map(weak_skill_score)
    d['is_mgmt'] = (
        d['profile_name'].astype(str).str.contains('|'.join(mgmt_tokens), regex=True)
        | d['core_skills'].astype(str).str.contains('|'.join(mgmt_tokens), regex=True)
    )

    low_weak = d[d['weak_skill_score'] >= 0.5]['salary_mid'].median()
    high_spec = d[d['weak_skill_score'] < 0.5]['salary_mid'].median()
    mgmt_med = d[d['is_mgmt']]['salary_mid'].median()
    non_mgmt_med = d[~d['is_mgmt']]['salary_mid'].median()

    chem = d[d['industry_group'] == '化学与化工'].sort_values('salary_mid', ascending=False)
    chem_text = ''
    if len(chem) >= 2:
        top = chem.iloc[0]
        second = chem.iloc[1]
        chem_text = (
            f'在化学与化工中出现反例：{top["profile_name"]}（约¥{top["salary_mid"]:,.0f}）'
            f'高于{second["profile_name"]}（约¥{second["salary_mid"]:,.0f}），说明管理标签并不必然对应更高薪资。'
        )

    exp_bins = d.dropna(subset=['exp_years']).copy()
    exp_text = ''
    if not exp_bins.empty:
        junior = exp_bins[exp_bins['exp_years'] <= 1]['salary_mid'].median()
        senior = exp_bins[exp_bins['exp_years'] >= 3]['salary_mid'].median()
        if pd.notna(junior) and pd.notna(senior):
            exp_text = f'经验维度上，1年及以下岗位薪资中位数约¥{junior:,.0f}，而3年及以上约¥{senior:,.0f}，呈现明显经验溢价。'

    scope_line = (
        '数据口径说明：对所有可以聚类的行业组进行了聚类（cluster_profiles.csv）；'
        '对样本量70以上的大样本行业组进行了回归分析（skill_impact.csv）；'
        '对样本量10-70的行业组进行了分组描述性统计（skill_value_robust.csv）。'
    )

    impact_line = ''
    if impact_path.is_file():
        try:
            imp = pd.read_csv(impact_path, low_memory=False)
            imp['coefficient'] = pd.to_numeric(imp.get('coefficient'), errors='coerce')
            imp['p_value'] = pd.to_numeric(imp.get('p_value'), errors='coerce')
            sig = imp[(imp['p_value'] < 0.1) & imp['coefficient'].notna()]
            if not sig.empty:
                top_pos = sig.sort_values('coefficient', ascending=False).head(1).iloc[0]
                top_neg = sig.sort_values('coefficient', ascending=True).head(1).iloc[0]
                impact_line = (
                    f'在70+样本行业回归中，{top_pos.get("industry_group","")}的“{top_pos.get("skill","")}”'
                    f'呈现显著正向薪资关联（系数约{top_pos.get("coefficient",0):.0f}）；'
                    '（注意：此处系数为回归系数，并非直接对应技能价值溢价）'
                    f'而{top_neg.get("industry_group","")}的“{top_neg.get("skill","")}”'
                    f'呈现显著负向关联（系数约{top_neg.get("coefficient",0):.0f}），'
                    '反映不同产业链对同类技能的定价差异。'
                )
        except Exception:
            impact_line = ''

    robust_line = ''
    if robust_path.is_file():
        try:
            rb = pd.read_csv(robust_path, low_memory=False)
            rb['pure_skill_value'] = pd.to_numeric(rb.get('pure_skill_value'), errors='coerce')
            rb = rb.dropna(subset=['pure_skill_value'])
            if not rb.empty:
                top_rb = rb.sort_values('pure_skill_value', ascending=False).head(1).iloc[0]
                low_rb = rb.sort_values('pure_skill_value', ascending=True).head(1).iloc[0]
                robust_line = (
                    f'在10-70样本行业的描述性统计中，{top_rb.get("industry_group","")}行业“{top_rb.get("skill","")}”'
                    f'的纯技能溢价约¥{top_rb.get("pure_skill_value",0):,.0f}，'
                    f'而{low_rb.get("industry_group","")}行业“{low_rb.get("skill","")}”约¥{low_rb.get("pure_skill_value",0):,.0f}，'
                    '说明中小样本行业内技能回报波动更大。'
                )
        except Exception:
            robust_line = ''

    lines = [
        scope_line,
        f'跨行业聚类结果显示，“技能描述偏泛化（如办公软件、沟通协作、基础销售）”的簇，其薪资中位数约¥{low_weak:,.0f}；'
        f'而“技能更专业化（如研发、技术支持、工艺或算法）”的簇约¥{high_spec:,.0f}，整体更高。',
        f'总体上，带有管理/经理/主管特征的簇薪资中位数约¥{mgmt_med:,.0f}，高于非管理簇的¥{non_mgmt_med:,.0f}。',
    ]
    if chem_text:
        lines.append(chem_text)
    if exp_text:
        lines.append(exp_text)
    if impact_line:
        lines.append(impact_line)
    if robust_line:
        lines.append(robust_line)
    return ''.join(lines)


def build_industry_salary_experience_figure():
    p = REPO_ROOT / 'regression_output' / 'exp_curve.csv'
    if not p.is_file():
        return ''
    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return ''
    req = {'industry_group', 'years_experience', 'predicted_salary'}
    if not req.issubset(df.columns):
        return ''

    d = df.copy()
    d['years_experience'] = pd.to_numeric(d['years_experience'], errors='coerce')
    d['predicted_salary'] = pd.to_numeric(d['predicted_salary'], errors='coerce')
    d = d.dropna(subset=['industry_group', 'years_experience', 'predicted_salary'])
    d = d[d['years_experience'].between(0, 10)]
    if d.empty:
        return ''

    # 按 0 年基准薪资排序，取前 8 个行业，避免图过挤
    base = d[d['years_experience'] == 0].groupby('industry_group', as_index=False)['predicted_salary'].mean()
    if base.empty:
        return ''
    keep = base.sort_values('predicted_salary', ascending=False).head(8)['industry_group']
    d = d[d['industry_group'].isin(keep)]

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans'],
        'axes.unicode_minus': False,
    })
    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=170)
    for ind, g in d.groupby('industry_group'):
        s = g.sort_values('years_experience')
        ax.plot(s['years_experience'], s['predicted_salary'], marker='o', linewidth=1.8, label=str(ind))

    ax.set_title('各行业经验-薪资关系（0-10年）', fontsize=12, fontweight='bold')
    ax.set_xlabel('经验年限')
    ax.set_ylabel('预测月薪（元）')
    ax.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False, title='行业')
    fig.tight_layout()

    out_dir = HERE / 'generated_figures'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'industry_salary_experience.png'
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path.resolve().as_uri()


def render_report_html(context):
    env = Environment(
        loader=FileSystemLoader(HERE),
        autoescape=select_autoescape(['html', 'xml']),
    )
    template = env.get_template(TEMPLATE_FILE.name)
    return template.render(context)


def generate_pdf(html_content):
    HTML(string=html_content, base_url=str(HERE)).write_pdf(str(OUTPUT_PDF))


def main():
    analysis_geography = (
        '地域就业机会呈现显著集聚特征，一线与强二线城市仍是岗位密度和高薪岗位的核心承载区。'
        '从就业机会视角看，城市层级与产业结构共同决定了岗位供给规模；从薪酬回报视角看，城市间平均薪资差异并不完全由岗位数量解释。'
        '因此，求职策略不宜仅按“岗位多寡”排序，而应同步评估岗位质量、薪资中枢与行业匹配度。'
        '对于高校就业指导而言，可结合区域产业链画像提供差异化去向建议，避免将“高薪城市”与“可进入城市”等同。'
    )
    analysis_education = (
        '学历要求与薪资水平总体呈正相关，但并非简单线性关系。'
        '在样本中，大专与本科层级构成主要岗位供给池，体现出应用型岗位和技术执行岗位的需求强度；'
        '硕博层级虽薪资更高，但岗位数量相对有限，且更集中于高门槛行业与研发导向岗位。'
        '这提示高校在课程设计中应同时强化基础能力、项目实践与可迁移技能，提升跨岗位适配能力。'
        '从人才供给侧看，单纯提升学历并不足以覆盖岗位分化风险，学历与技能组合的协同构建才是提升就业质量的关键。'
    )
    analysis_industry = (
        '按岗位规模看，当前样本需求最集中的行业包括：机械工程（261）、计算机技术（186）、人力资源（154）、'
        '电子信息（150）、材料与轻工（110）。在样本量不少于50的行业中，平均薪资领先行业为：'
        '计算机技术（¥12,452）、能源与动力（¥12,319）、电子信息（¥11,722）、生物与医药（¥11,627）、'
        '机械工程（¥10,992）。这说明当前劳动力需求既呈现“制造/工程等大体量行业吸纳岗位”，也呈现“数字技术与高壁垒行业'
        '拉高薪资中枢”的双轨特征。进一步看，岗位数量与薪资水平并不同步：机械工程岗位规模最大，反映实体产业链对工程实施'
        '、运维与交付的持续需求；计算机技术则在规模与薪资上同时领先，体现数字化岗位需求广且溢价高。'
        '对求职者而言，这种结构意味着“稳就业”与“高回报”可能对应不同赛道：前者偏向岗位容量更大的'
        '行业，后者偏向技术门槛高、业务复杂度高的岗位族群。'
    )
    analysis_industry_exp_skill = (
        '图4 各行业经验-薪资关系（0-10年）展示的是大样本行业组（70个以上）的回归分析结果。经验曲线覆盖8个行业，'
        '0-5年薪资增速较快行业为：人力资源（0-5年约+167.9%）、电子信息（0-5年约+163.3%）、商业与贸易'
        '（0-5年约+132.0%）。技能价值层面，大样本回归覆盖8个行业，中小样本稳健估计覆盖9个行业；稳健溢价示例包括：'
        '金融与经济-Vue（约¥20,000）、金融与经济-Redis（约¥20,000）、金融与经济-金融工程（约¥15,000）。'
        '这组结果表明，行业薪资差距并非仅由经验年限决定，技能组合与行业情境的交互作用同样关键。具体而言，经验曲线反映的是'
        '“同一行业内部的人力资本累积回报”，其斜率差异揭示了行业晋升机制与岗位分层速度的不同；而技能溢价结果反映的是'
        '“特定技能在特定产业场景中的边际定价”。当两者叠加时，会形成显著的路径分化：在高增速行业中，若技能组合贴合核心业务'
        '链条，薪资提升通常更早出现且更陡峭；在增速较缓行业中，即便经验增加，若缺乏关键技能包，收入提升也可能趋于平缓。'
        '因此，职业策略不应只看工作年限，而应同步评估目标行业的经验回报曲线和关键技能清单，优先投资“可跨岗位迁移且具行业'
        '稀缺性”的技能组合，以提升中长期薪资弹性。'
    )

    figures, industry_overview_src, cluster_example_src, cluster_skill_analysis = load_report_figures(
        analysis_geography,
        analysis_education,
        analysis_industry,
    )
    discussion_paragraphs = [
        '结合本次多维结果可见，行业间差异并不只体现在“起薪高低”，更体现在“成长斜率”和“技能回报结构”上：'
        '同一学历门槛下，不同行业对经验累积的定价不同；同一技能标签在不同产业链中的溢价也可能显著分化。\n'
        '\t将地域、学历与行业三章串联，可把就业质量概括为“机会密度（城市/行业岗位池）—门槛结构（学历与经验）'
        '—能力定价（技能簇与行业情境）”的共同作用；高校宜将培养方案与产业链任务对齐，推动学生从“知道”'
        '走向“能实践，会运用”。\n'
        '\t对高校与培训体系，教学供给需从“通用能力堆叠”转向“行业场景能力包”，以真实岗位任务为主线，'
        '将基础素养、工具能力与项目实践纳入可复核的证据链。对求职者，应以目标行业高回报技能簇为核心，'
        '关注地域产业结构与岗位分层，避免跨行业平均用力造成的错配。对企业，建议在岗位描述中写明可核验'
        '的能力要素（项目成果、工具栈、业务指标等），以降低匹配摩擦。\n'
        '\t各学院可在“行业画像—能力标准—课程地图—实习评价”闭环下分类施策：理工强化工程实践、数据/仿真工具链'
        '与文档复盘；计算机与信息类补足业务数据、可靠性与安全合规等可上线能力；生命科学与医药类按研发、'
        '技术支持与市场沟通等岗位簇组织案例课；经管与人文社科强化可量化产出与真实项目；艺术设计与传媒对齐'
        '作品集与迭代交付；外语绑定涉外场景与行业术语；职业教育与应用型以岗位任务为单元并推行双导师评价。'
        '学校层面可建设跨学院微专业/荣誉项目与就业数据中台，将招聘高频技能映射到课程标签并年度修订。',
    ]
    html = render_report_html({
        'report_title': f'{datetime.now().year}年春季国内就业市场洞察：\n基于网络招聘数据的多维分析与建模',
        'report_subtitle': '面向高校人才培养与行业供需的跨维度洞察与行动建议',
        'author_names': 'CareerMind 就业研究院',
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'overview': (
            f'本报告基于 CareerMind 数据平台自前程无忧（51Job，入口域名为 51job.com，亦常见 51jobs.com 指向同一招聘服务体系）'
            f'采集、经清洗后在分析管线中可用的岗位样本，共 {REPORT_ANALYSIS_RECORDS} 条有效记录；'
            '旨在从显性岗位需求和技能画像角度，刻画当前样本内跨地域、跨学历与跨行业的结构性特征。'
            '由于数据来源于公开招聘渠道，本文采用定量分析与文本挖掘相结合的方法，探索院校、技能、地域等核心维度的聚集与差异。'
            '需要强调的是，本研究侧重于“可见”能力要求与岗位标签，力求在现有数据框架下提供稳健的行业洞察，'
            '而不试图覆盖所有隐性就业机制。本文结论主要反映平台内招聘生态，对于外部渠道或非公开招聘行为的解释仍需谨慎。'
            '在研究设计上，报告同时关注“岗位数量—能力要求—薪酬回报”三条主线，'
            '并通过跨维度对照识别就业结构中的共性规律与差异模式，以避免单指标解读造成的偏差。'
            '面向应用端，本报告将分析结果转化为高校培养、学生求职与企业招聘三类主体可执行的建议框架，'
            '强调从数据证据出发进行路径选择，而非仅凭经验判断。'
        ),
        'keywords': '就业市场分析；招聘数据；岗位地理分布；学历与薪资；行业聚类；经验-薪资建模',
        'introduction': (
            '在经济结构持续转型与产业升级加速推进的背景下，就业市场正呈现显著的动态性与结构性变化。'
            '传统就业分析较多依赖宏观统计或问卷数据，在时效性、颗粒度和可复核性方面存在局限。'
            '随着招聘平台数据可得性提高，岗位文本、薪资区间、学历门槛、经验要求和行业属性为微观就业结构刻画提供了新的证据基础。'
            '基于此，本项目构建“聚类画像 + 行业建模”的双层框架：前者刻画行业内部岗位类型及技能组合，后者量化经验与技能对薪资的作用。'
            '研究目标是从数据驱动角度识别结构性机会与约束，为高校培养方案、求职路径设计与企业岗位策略提供可执行参考。'
            '与传统描述性统计不同，本研究进一步引入岗位文本解析与行业内分层建模，'
            '在同一分析框架下统一处理“岗位属性、技能标签、经验要求与薪资表现”之间的联动关系。'
            '方法上，项目采用“数据清洗—特征构造—行业聚类—薪资建模—结果解释”的流水线，'
            '并通过可视化与文字报告双通道输出结果，以兼顾决策效率与结论透明度。'
            '从问题导向看，报告重点回答三类核心问题：不同城市与行业的岗位机会如何分化；'
            '学历与技能在不同岗位场景中如何共同影响薪资；以及在给定行业目标下应如何配置学习投入与求职策略。'
        ),
        'analysis_geography': analysis_geography,
        'analysis_education': analysis_education,
        'analysis_industry': analysis_industry,
        'analysis_industry_exp_skill': analysis_industry_exp_skill,
        'discussion_paragraphs': discussion_paragraphs,
        'discussion': '\n\n'.join(discussion_paragraphs),
        'limitations': (
            '本研究数据主要来自公开招聘平台，存在平台样本偏倚：部分企业在平台发布更活跃，另一些企业则更偏向官网或内推渠道。'
            '尽管报告通过行业内聚类与分行业回归提升了结构解释力，但证据基础仍是公开招聘文本，'
            '难以覆盖企业内推、校友网络、熟人关系与岗位隐性门槛等非公开机制；'
            '因此，本报告结论更适用于解释“显性岗位需求市场”，而不应被直接外推为全部就业通道的完备刻画。'
            '在当前样本中，聚类结果里仍可见多组小样本画像簇，其更适合作为定性线索与假设生成来源，'
            '不宜单独支撑强统计推断；城市字段亦存在“城市—区县”混合粒度，可能放大地域对比的口径差异。'
            '此外，部分技能溢价极值还可能受到样本稀疏、岗位标题噪声与行业内岗位异质性的共同影响，'
            '从而抬高局部估计的不确定性。后续可通过跨平台数据融合、时间滚动验证与更严格的因果识别策略，'
            '进一步提升外推性与稳健性。'
        ),
        'conclusion': (
            '综合来看，本期就业市场呈现“城市集聚、行业分化、学历分层、技能异质回报”并存格局，'
            '且行业内部仍存在显著的岗位簇分化与经验—技能联动效应。'
            '研究成果的核心启示在于：显性招聘市场更奖励“可被验证的岗位能力”，而非笼统的素质标签；'
            '高校若要把就业竞争力做实，需要把培养目标从“知识覆盖”转向“能力证据 + 行业语境”。'
            '求职侧建议围绕目标行业构建技能组合并关注区域与产业链匹配；'
            '高校侧建议强化与产业链协同的课程模块与项目实践，将培养路径与“行业场景能力包”对齐；'
            '用人侧建议在岗位描述中提高能力标签的可验证性和透明度，以降低匹配摩擦并提升招聘效率。'
            '对各学院的具体行动建议可概括为：一是每学期选取若干目标行业做“岗位簇—技能簇”对照表，'
            '反向修订核心课与集中实践环节；二是把毕业论文/毕业设计与可公开作品、竞赛、企业课题挂钩，'
            '形成可展示成果；三是建立实习质量量规（任务复杂度、独立负责度、协作记录），避免实习流于形式；'
            '四是推动教师发展计划纳入产业研修与真实案例库建设，使课堂语言与用人方语言逐步同构。'
            '若上述措施能制度化推进，学校将在不牺牲学术深度的前提下，显著提升学生进入显性劳动力市场的'
            '匹配效率与起薪—成长曲线的稳健性。'
        ),
        'references': [
            'OECD. Education at a Glance 2025. Paris: OECD Publishing.',
            'World Bank. World Development Report 2026. Washington, DC: World Bank.',
            'Autor, D. H., Levy, F., & Murnane, R. J. (2003). The Skill Content of Recent Technological Change. Quarterly Journal of Economics, 118(4), 1279–1333.',
            'Acemoglu, D., & Autor, D. H. (2011). Skills, Tasks and Technologies: Implications for Employment and Earnings. In Handbook of Labor Economics, Vol. 4B, 1043–1171.',
            'Card, D. (1999). The Causal Effect of Education on Earnings. In Handbook of Labor Economics, Vol. 3, 1801–1863.',
            'Deming, D. J. (2017). The Growing Importance of Social Skills in the Labor Market. Quarterly Journal of Economics, 132(4), 1593–1640.',
            'Grimmer, J., & Stewart, B. M. (2013). Text as Data: The Promise and Pitfalls of Automatic Content Analysis Methods for Political Texts. Political Analysis, 21(3), 267–297.',
            'International Labour Organization (ILO). Key Indicators of the Labour Market (KILM) / Global Employment Trends (系列报告，用于宏观劳动力市场对照).',
            '前程无忧（51Job）公开职位页面与平台说明文档（企业发布职位、求职者检索与投递的产品形态描述，用于界定数据来源性质）。',
            '国家统计局. 《中国劳动统计年鉴》（历年；用于薪资、就业等宏观口径与本研究微观样本的对照阅读）。',
            '教育部、人力资源和社会保障部等发布的普通高校毕业生就业创业工作政策文件（用于高校人才培养与就业政策的制度背景）。',
            'CareerMind 项目内部数据字典、ETL 与建模说明（job_vec.csv、cluster_profiles.csv、skill_impact.csv、skill_value_robust.csv 等产出定义）。',
        ],
        'appendix_methodology': (
            '【数据概况】本研究原始岗位文本与结构化字段来自前程无忧网络招聘平台（51Job；主站入口为 https://www.51job.com/，'
            '用户与文档中亦常写作 51jobs.com 等变体，均指向同一招聘服务体系）。前程无忧成立于 1998 年，是国内较早开展综合性网络招聘与人力资源服务的机构之一，'
            '面向企业端提供职位发布与人才寻访等服务，面向求职者提供职位检索、在线投递与校招信息等，行业与职类覆盖面广，'
            '适合作为观测“显性招聘需求”与岗位文本特征的窗口；同时需注意其岗位池受平台用户结构、企业发布习惯与审核规则影响，'
            '不等同于全国劳动力市场的完整抽样。经项目内去重、字段校验与清洗规则过滤后，本报告各章统计、聚类与回归分析所基于的有效岗位样本量为 '
            f'{REPORT_ANALYSIS_RECORDS} 条；单条记录通常包含职位名称、薪资区间、工作地点、学历与经验要求、技能或任职要求文本等，'
            '并映射至统一的行业分组与可计算特征，以支持地理分布、学历—薪资、行业聚类与薪资建模等模块。'
            '【方法论与流水线】项目流程为：自平台抓取得到 jobs().csv 原始表 → temp.py 抽取 other_requirement 等字段形成 jobs(1).csv → '
            'etl.py 完成经验年限数字化、技能共线合并、薪资口径统一与向量化，得到 job_vec.csv，并输出 skill_merge_preview.csv；'
            '随后 job_clustering.py 在行业维度内生成聚类画像（cluster_profiles.csv），salary_regression.py 分行业估计经验—薪资曲线与技能系数'
            '（如 skill_impact.csv、skill_value_robust.csv 等）。可视化与报告由 dashboard 与 generate_report.py 汇总呈现。详情见项目内 Code 与 Data 说明。'
        ),
        'skill_merge_appendix_note': (
            f'下列为 ETL 阶段 compute_skill_merge_preview 输出的技能合并建议：在行业内部基于技能 0/1 共现矩阵计算 Pearson 相关系数，'
            f'当 r > {SKILL_MERGE_CORRELATION_THRESHOLD_DOC} 时合并相关技能，以组内出现频次最高的技能作为代表标签。'
        ),
        'figures': figures,
        'industry_overview_src': industry_overview_src,
        'cluster_example_src': cluster_example_src,
        'cluster_skill_analysis': cluster_skill_analysis,
        'skill_merge_rows': load_skill_merge_preview_rows(),
    })
    generate_pdf(html)
    print(f'PDF_OUTPUT:{OUTPUT_PDF.resolve()}')


if __name__ == '__main__':
    main()
