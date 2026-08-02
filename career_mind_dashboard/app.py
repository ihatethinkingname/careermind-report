import html
import json
import os
import sys
import time
from typing import Optional

import base64
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# ensure local package import works when running from project root
sys.path.append(os.path.dirname(__file__))
from data_bridge import (
    load_jobs,
    extract_education_labels,
    get_overview_stats,
    get_industries,
    get_industries_from_cluster_profiles,
    get_clusters_for_industry,
    get_skill_importance,
    get_exp_curve,
    load_report_sections,
    load_skill_merge_preview,
    files_status,
)
from generate_pdf import get_pdf_path
import chart_insights

CODE_REPO_URL = "https://github.com/ihatethinkingname/careermind-report/tree/main/code"
DATA_REPO_URL = "https://github.com/ihatethinkingname/careermind-report/tree/main/data"


st.set_page_config(page_title='CareerMind Dashboard', layout='wide', initial_sidebar_state='collapsed')

# simple dark / Morandi-like accent stylesheet (lightweight)
st.markdown(
    """
    <style>
    .stApp { background:#0f1720; color:#e6eef6 }
    /* 标题距页面顶部：改这里 padding-top（整块主内容起点，含标题） */
    .block-container { padding-top: 7rem; }
    /* 主区略减左右内边距，四指标更贴近页面两侧（与 layout=wide 配合） */
    div[data-testid='stMain'] > div > div.block-container {
      max-width: 100%;
      padding-left: 0.5rem;
      padding-right: 0.5rem;
    }
    .metric-label { color: #cbd5e1 }
    /* 顶部概览四个 st.metric：标签与数值居中（与 Paper 区视觉一致） */
    [data-testid="stMetric"] { text-align: center; }
    [data-testid="stMetric"] > div { justify-content: center !important; align-items: center !important; }
    .card { background:#111827; padding:12px; border-radius:8px; }
    /* 工具栏图标：去掉 Streamlit 给 st.markdown 套的浅灰边框 */
    div[data-testid="stMarkdown"]:has(.careermind-toolbar-icon) {
      border: none !important;
      outline: none !important;
      box-shadow: none !important;
      background: transparent !important;
    }
    div[data-testid="stMarkdown"]:has(.careermind-toolbar-icon) > div {
      border: none !important;
      padding: 0 !important;
      background: transparent !important;
    }
    .careermind-toolbar-icon img {
      border: none !important;
      outline: none !important;
      box-shadow: none !important;
      vertical-align: middle;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.careermind-toolbar-icon) {
      border: none !important;
      box-shadow: none !important;
    }
    /* 行业穿透：聚类卡片内「核心技能」行高与下方经验/其他要求 */
    .cluster-core-skills {
      line-height: 1.72;
      font-size: 0.95rem;
      color: #e6eef6;
      margin: 0 0 2px 0;
    }
    .cluster-field-label {
      color: #94a3b8;
      font-size: 0.78rem;
      font-weight: 600;
      letter-spacing: 0.02em;
      margin: 10px 0 4px 0;
    }
    .cluster-field-body {
      line-height: 1.58;
      font-size: 0.88rem;
      color: #cbd5e1;
      margin: 0;
    }
    /*
     * 聚类底部说明与上方的间距：请改 padding-top。
     * 不要用 <p> + margin-top：Streamlit 对 markdown 内首个 p 常设 margin-top:0，会盖掉你的 margin-top；
     * margin-bottom 往往仍生效，所以会出现「改上边距没用、改下边距有用」的现象。
     */
    .cluster-sample-hint {
      padding-top: 1.25rem;
      margin: 0.5rem;
      color: #94a3b8;
      font-size: 0.875rem;
      line-height: 1.45;
    }
    .report-paragraph {
      text-indent: 2em;
      line-height: 1.9;
      margin: 0.6rem 0 1rem 0;
      color: #dbe5f3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_currency(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return '—'
    return f"¥{x:,.0f}"


def _deepseek_api_key() -> str:
    k = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if k:
        return k
    try:
        return str(st.secrets.get('DEEPSEEK_API_KEY', '') or '').strip()
    except Exception:
        return ''


def _render_chart_readme(
    chart_title: str,
    local_md: str,
    llm_df: Optional[pd.DataFrame],
    chart_id: str,
    use_llm: bool,
    api_key: str,
) -> None:
    """每张图下：本地读图说明 + 可选 DeepSeek（结果按数据摘要缓存于 session_state）。"""
    st.markdown('##### 读图说明')
    st.markdown(local_md)
    if not (use_llm and api_key and llm_df is not None and not llm_df.empty):
        return
    dig = chart_insights.digest_for_df(llm_df, chart_id)
    cache_key = f'cm_llm_{chart_id}_{dig}'
    if cache_key not in st.session_state:
        snippet = chart_insights.df_snippet(llm_df, max_rows=18, max_chars=3200)
        with st.spinner('DeepSeek 补充说明中…'):
            out = chart_insights.deepseek_chart_note(chart_title, local_md, snippet, api_key)
        st.session_state[cache_key] = (out or '').strip()
    extra = st.session_state.get(cache_key, '')
    if extra:
        st.markdown('**DeepSeek 补充**\n\n' + extra)
    else:
        st.caption('DeepSeek 未返回有效内容（请检查密钥、网络或 API 额度）。')


def _find_icon_path(filename: str) -> Optional[str]:
    # 优先在 career_mind_dashboard/asset 目录中查找
    asset_dir = os.path.join(os.path.dirname(__file__), 'asset')
    asset_path = os.path.join(asset_dir, filename)
    if os.path.exists(asset_path):
        return asset_path
    
    # 回退到 analysis 根目录
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(parent, filename)
    return path if os.path.exists(path) else None


def _file_to_data_uri(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.svg': 'image/svg+xml',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
    }.get(ext, 'application/octet-stream')
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('ascii')
    return f'data:{mime};base64,{b64}'


def _resolve_pdf_path() -> Optional[str]:
    """Resolve a usable PDF path with a fallback re-check."""
    p = get_pdf_path()
    if p and os.path.isfile(p):
        return p
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    candidates = [
        os.path.join(os.path.dirname(__file__), 'CareerMind_Report.pdf'),
        os.path.join(parent, 'CareerMind_Report.pdf'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


@st.cache_data(show_spinner=False)
def _pdf_href_from_path(path: str, mtime: float) -> Optional[str]:
    """Build a stable data URI for PDF preview link."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, 'rb') as f:
            pdf_b64 = base64.b64encode(f.read()).decode('ascii')
        return f'data:application/pdf;base64,{pdf_b64}'
    except Exception:
        return None


def _get_ready_pdf_href() -> Optional[str]:
    """Return a robust PDF href with retries and session cache fallback."""
    path = _resolve_pdf_path()
    if not path or not os.path.isfile(path):
        return st.session_state.get("cm_pdf_href")

    size = os.path.getsize(path)
    mtime = os.path.getmtime(path)
    sig = f"{path}|{mtime}|{size}"
    cached_sig = st.session_state.get("cm_pdf_href_sig")
    if cached_sig == sig:
        return st.session_state.get("cm_pdf_href")

    href = None
    # PDF 可能刚生成完成但文件句柄尚未稳定，短暂重试可避免“首开空白页”。
    for _ in range(5):
        path = _resolve_pdf_path()
        if not path or not os.path.isfile(path):
            time.sleep(0.2)
            continue
        if os.path.getsize(path) < 1024:
            time.sleep(0.2)
            continue
        try:
            with open(path, "rb") as f:
                head = f.read(5)
            if head != b"%PDF-":
                time.sleep(0.2)
                continue
            href = _pdf_href_from_path(path, os.path.getmtime(path))
            if href:
                size = os.path.getsize(path)
                mtime = os.path.getmtime(path)
                sig = f"{path}|{mtime}|{size}"
                break
        except Exception:
            time.sleep(0.2)

    if href:
        st.session_state["cm_pdf_href"] = href
        st.session_state["cm_pdf_href_sig"] = sig
        return href
    return st.session_state.get("cm_pdf_href")


def _paper_pdf_b64_from_href(href: Optional[str]) -> Optional[str]:
    """从 data: PDF href 取出纯 base64；不含 data: 前缀（Edge 禁止顶层导航到 data: URL）。"""
    if not href or not href.startswith("data:application/pdf;base64,"):
        return None
    return href[28:]


def _paper_pdf_viewer_shell_chunks() -> list:
    """新标签页内嵌预览壳 HTML 分片（join 后为完整 document）。"""
    return [
        "<!DOCTYPE html><html><head><meta charset=utf-8><title>CareerMind Report</title><style>",
        "html,body{margin:0;height:100%;background:#0f172a;font-family:system-ui,sans-serif;"
        "display:flex;flex-direction:column;overflow:hidden;}",
        ".hdr{flex:0 0 auto;background:#0f172a;border-bottom:1px solid #334155;}",
        ".track{height:4px;background:#1e293b;margin:0 16px 8px 16px;border-radius:2px;overflow:hidden;}",
        ".fill{height:100%;width:0;border-radius:2px;background:linear-gradient(90deg,#60a5fa,#a78bfa,#f472b6);transition:width .2s ease-out;}",
        ".lab{color:#94a3b8;text-align:center;font-size:13px;padding:8px 12px 4px;}",
        "#box{flex:1 1 auto;min-height:0;position:relative;background:#111827;}",
        "iframe{border:0;position:absolute;inset:0;width:100%;height:100%;display:block;background:#fff;}",
        "</style></head><body><div class=hdr><div class=lab id=L>正在加载 PDF…</div><div class=track><div class=fill id=B></div></div></div><div id=box></div>",
        "<script>",
        "(function(){var g=null;try{g=window.opener&&window.opener.__cm_pdf_b64_pending;}catch(y){}",
        'if(!g){try{g=localStorage.getItem("__cm_pdf_b64_pending");localStorage.removeItem("__cm_pdf_b64_pending");}catch(_){}}',
        "try{if(window.opener)delete window.opener.__cm_pdf_b64_pending;}catch(_){}",
        'var B=document.getElementById("B");var L=document.getElementById("L");var box=document.getElementById("box");',
        'var p=0,t=setInterval(function(){p=Math.min(p+5,88);if(B)B.style.width=p+"%";},90);',
        'function F(){clearInterval(t);if(L)L.textContent="无法加载 PDF，请关闭本页后重试。";}',
        "if(!g){F();return;}",
        "setTimeout(function(){try{",
        "var z=atob(g),n=z.length,a=new Uint8Array(n),i;for(i=0;i<n;i++)a[i]=z.charCodeAt(i);",
        'var o=new Blob([a],{type:"application/pdf"});var u=URL.createObjectURL(o);',
        'clearInterval(t);if(B)B.style.width="100%";',
        'setTimeout(function(){if(L)L.style.display="none";var H=document.querySelector(".hdr");if(H)H.style.display="none";},220);',
        'var I=document.createElement("iframe");I.title="PDF";I.src=u;box.appendChild(I);',
        "setTimeout(function(){URL.revokeObjectURL(u);},300000);",
        "}catch(e){F();}},0);})();",
        "</script></body></html>",
    ]


def _render_paper_pdf_toolbar_component(href: str, paper_img_uri: Optional[str]) -> None:
    """Paper 入口：必须用 components.html（原生 DOM），勿在 st.markdown 里写 onclick。

    Streamlit Markdown 走 React 时会将字符串型 onClick 视为非法（React #231）。
    """
    b64 = _paper_pdf_b64_from_href(href)
    if not b64:
        return
    chunks_json = json.dumps(_paper_pdf_viewer_shell_chunks(), ensure_ascii=False)
    chunks_b64 = base64.b64encode(chunks_json.encode("utf-8")).decode("ascii")
    b64_js = json.dumps(b64, ensure_ascii=True)
    chunks_b64_js = json.dumps(chunks_b64, ensure_ascii=True)
    img_html = ""
    if paper_img_uri:
        img_html = (
            '<img src="'
            + html.escape(paper_img_uri, quote=True)
            + '" width="72" alt="Paper" style="display:block;margin:0 auto 6px;border:none;"/>'
        )
    script = """(function(){
  var b64 = __B64__;
  var chunksB64 = __CHUNKS_B64__;
  var btn = document.getElementById("cm-paper-btn");
  if(!btn||!b64)return;
  var P = JSON.parse(atob(chunksB64));
  btn.addEventListener("click",function(ev){
    ev.preventDefault();
    try{window.__cm_pdf_b64_pending=b64;}catch(e0){}
    try{localStorage.setItem("__cm_pdf_b64_pending",b64);}catch(e1){}
    var S=P.join("");
    var shellBlob=new Blob([S],{type:"text/html;charset=utf-8"});
    var shellUrl=URL.createObjectURL(shellBlob);
    var w=window.open(shellUrl,"_blank");
    if(!w){window.location.href=shellUrl;}
    setTimeout(function(){try{URL.revokeObjectURL(shellUrl);}catch(z){}},60000);
  });
})();""".replace(
        "__B64__", b64_js
    ).replace(
        "__CHUNKS_B64__", chunks_b64_js
    )
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<style>
html,body{{margin:0;padding:0;background:transparent;overflow:hidden;}}
a#cm-paper-btn{{display:inline-block;text-decoration:none;color:#e6eef6;font-weight:600;
  text-align:center;width:100%;cursor:pointer;font-family:system-ui,sans-serif;}}
</style></head>
<body>
<div class="careermind-toolbar-icon" style="text-align:center;border:none;padding:0;margin:0">
<a href="#" id="cm-paper-btn">{img_html}Paper</a>
</div>
<script>{script}</script>
</body></html>"""
    components.html(doc, height=118, width=200, scrolling=False)


def _normalize_city_name(raw_city: str) -> str:
    """从 city 字段提取用于坐标匹配的城市名。

    支持「上海·浦东新区」（取地级市）与「湖北·黄冈」（取右侧区县/市）等写法：
    若含间隔符，从右向左找第一个出现在坐标表中的分段；否则再退回左侧并去掉「市」后缀。
    """
    city = str(raw_city).strip()
    if not city:
        return ''
    coords = _city_coord_map()
    for sep in ('·', '•', '|'):
        if sep in city:
            parts = [p.strip() for p in city.split(sep) if p.strip()]
            for p in reversed(parts):
                cand = p.replace('省', '').replace('市', '').strip()
                if cand in coords:
                    return cand
            city = parts[0]
            break
    return city.replace('市', '').strip()


def _city_coord_map():
    return {
        '北京': (39.9042, 116.4074), '上海': (31.2304, 121.4737), '广州': (23.1291, 113.2644),
        '深圳': (22.5431, 114.0579), '杭州': (30.2741, 120.1551), '南京': (32.0603, 118.7969),
        '苏州': (31.2989, 120.5853), '无锡': (31.4912, 120.3119), '常州': (31.8107, 119.9737),
        '宁波': (29.8683, 121.5440), '嘉兴': (30.7461, 120.7555), '绍兴': (30.0303, 120.5802),
        '湖州': (30.8943, 120.0868), '台州': (28.6564, 121.4208), '温州': (27.9949, 120.6994),
        '金华': (29.0791, 119.6474), '合肥': (31.8206, 117.2290), '武汉': (30.5928, 114.3055),
        '长沙': (28.2282, 112.9388), '南昌': (28.6820, 115.8579), '福州': (26.0745, 119.2965),
        '厦门': (24.4798, 118.0894), '泉州': (24.8741, 118.6759), '济南': (36.6512, 117.1201),
        '青岛': (36.0671, 120.3826), '烟台': (37.4638, 121.4479), '天津': (39.3434, 117.3616),
        '重庆': (29.5630, 106.5516), '成都': (30.5728, 104.0668), '西安': (34.3416, 108.9398),
        '郑州': (34.7472, 113.6249), '沈阳': (41.8057, 123.4315), '大连': (38.9140, 121.6147),
        '长春': (43.8171, 125.3235), '哈尔滨': (45.8038, 126.5349), '昆明': (25.0389, 102.7183),
        '贵阳': (26.6470, 106.6302), '南宁': (22.8170, 108.3669), '海口': (20.0440, 110.1999),
        '石家庄': (38.0428, 114.5149), '太原': (37.8706, 112.5489), '兰州': (36.0611, 103.8343),
        '乌鲁木齐': (43.8256, 87.6168), '呼和浩特': (40.8426, 111.7492), '拉萨': (29.6520, 91.1721),
        '银川': (38.4872, 106.2309), '香港': (22.3193, 114.1694), '澳门': (22.1987, 113.5439),
        '东莞': (23.0207, 113.7518), '佛山': (23.0215, 113.1214), '惠州': (23.1115, 114.4168),
        '珠海': (22.2710, 113.5767), '中山': (22.5176, 113.3928), '昆山': (31.3854, 120.9808),
        '义乌': (29.3069, 120.0759), '太仓': (31.4526, 121.1090), '湘西': (28.3113, 109.7389),
        # 常见地级/县级市（招聘数据中高频、原表未覆盖）
        '南通': (32.0147, 120.8576), '芜湖': (31.3348, 118.4331), '岳阳': (29.3572, 113.1287),
        '襄阳': (32.0424, 112.1441), '江门': (22.5787, 113.0815), '常熟': (31.6544, 120.7485),
        '保定': (38.8673, 115.4845), '衡阳': (26.8932, 112.5720), '清远': (23.6818, 113.0563),
        '镇江': (32.1896, 119.4551), '徐州': (34.2044, 117.2858), '洛阳': (34.6197, 112.4540),
        # 补充地级/县级市（来自 job_vec 中曾未映射的高频地名）
        '肇庆': (23.0472, 112.4655), '盐城': (33.3474, 120.1636), '天门': (30.6633, 113.1669),
        '酒泉': (39.7337, 98.4944), '株洲': (27.8270, 113.1339), '咸阳': (34.3336, 108.7093),
        '潮州': (23.6567, 116.6226), '邢台': (37.0706, 114.5049), '韶关': (24.8104, 113.5972),
        '邯郸': (36.6256, 114.4907), '荆州': (30.3348, 112.2387), '滁州': (32.3016, 118.3168),
        '吉安': (27.1138, 114.9928), '十堰': (32.6294, 110.7989), '廊坊': (39.5186, 116.7036),
        '眉山': (30.0771, 103.8485), '郴州': (25.7705, 113.0147), '广安': (30.4554, 106.6334),
        '赣州': (25.8318, 114.9335), '桂林': (25.2736, 110.2900), '连云港': (34.5967, 119.2216),
        '漳州': (24.5129, 117.6762), '张家口': (40.8119, 114.8863), '阜阳': (32.8969, 115.8197),
        '开封': (34.7971, 114.3074), '许昌': (34.0357, 113.8526), '舟山': (29.9853, 122.2072),
        '承德': (40.9730, 117.9392), '临沂': (35.1041, 118.3564), '黔东南': (26.5836, 107.9775),
        '榆林': (38.2854, 109.7346), '菏泽': (35.2336, 115.4807), '运城': (35.0264, 111.0075),
        '泰安': (36.2001, 117.0889), '湘潭': (27.8295, 112.9441), '乌兰察布': (41.0340, 113.1145),
        '德阳': (31.1269, 104.3980), '黄石': (30.2000, 115.0389), '六盘水': (26.5927, 104.8304),
        '锦州': (41.0951, 121.1270), '南阳': (32.9907, 112.5283), '湛江': (21.2707, 110.3594),
        '鄂州': (30.3919, 114.8949), '淄博': (36.8135, 118.0549), '汕头': (23.3535, 116.6820),
        '漯河': (33.5815, 114.0168), '滨州': (37.3819, 118.0249), '贺州': (24.4142, 111.5526),
        '邵阳': (27.2389, 111.4677), '达州': (31.2090, 107.4679), '河源': (23.7436, 114.6978),
        '潍坊': (36.7069, 119.1618), '塔城': (46.7456, 82.9803), '唐山': (39.6309, 118.1802),
        '抚州': (27.9492, 116.3582), '泸州': (28.8717, 105.4419), '鞍山': (41.1083, 122.9946),
        '曲靖': (25.4900, 103.7962), '南充': (30.8373, 106.1107), '本溪': (41.2942, 123.7669),
        '宿迁': (33.9630, 118.2752), '三亚': (18.2528, 109.5119), '威海': (37.5133, 122.1214),
        '柳州': (24.3255, 109.4286), '玉林': (22.6540, 110.1809), '莆田': (25.4541, 119.0078),
        '宁德': (26.6617, 119.5272), '三明': (26.2650, 117.6390), '龙岩': (25.0916, 117.0179),
        '宣城': (30.9457, 118.7588), '马鞍山': (31.6894, 118.5079), '淮南': (32.6255, 116.9999),
        '蚌埠': (32.9407, 117.3632), '安庆': (30.5435, 117.0636), '铜陵': (30.9456, 117.8121),
        '池州': (30.6648, 117.4895), '亳州': (33.8693, 115.7789), '淮北': (33.9548, 116.7983),
        '宿州': (33.6339, 116.9784), '聊城': (36.4560, 115.9855), '日照': (35.4167, 119.5269),
        '德州': (37.4513, 116.3595), '东营': (37.4336, 118.6746), '莱芜': (36.2138, 117.6767),
        '葫芦岛': (40.7110, 120.8369), '营口': (40.6669, 122.2354), '盘锦': (41.1199, 122.0707),
        '丹东': (40.0005, 124.3544), '辽阳': (41.2694, 123.1815), '铁岭': (42.2237, 123.7260),
        '朝阳': (41.5737, 120.4504), '四平': (43.1664, 124.3508), '松原': (45.1411, 124.8253),
        '通化': (41.7284, 125.9397), '白山': (41.9408, 126.4244), '延边': (42.9047, 129.5089),
        '齐齐哈尔': (47.3543, 123.9182), '大庆': (46.5877, 125.1031), '佳木斯': (46.7999, 130.3188),
        '牡丹江': (44.5513, 129.6332), '绥化': (46.6374, 126.9690), '黑河': (50.2451, 127.5286),
        '鸡西': (45.2952, 130.9693), '鹤岗': (47.3499, 130.2979), '双鸭山': (46.6469, 131.1591),
        '伊春': (47.7275, 128.8994), '七台河': (45.7713, 131.0031), '大兴安岭': (51.6731, 124.7105),
        '石河子': (44.3059, 86.0419), '昌吉': (44.0146, 87.3040), '库尔勒': (41.7259, 86.1746),
        '阿克苏': (41.1688, 80.2606), '喀什': (39.4704, 75.9897), '和田': (37.1142, 79.9222),
        '伊犁': (43.9169, 81.3241), '哈密': (42.8185, 93.5150), '克拉玛依': (45.5798, 84.8892),
        '广元': (32.4337, 105.8298), '遂宁': (30.5328, 105.5929), '内江': (29.5802, 105.0584),
        '乐山': (29.5522, 103.7657), '宜宾': (28.7519, 104.6308), '自贡': (29.3390, 104.7784),
        '攀枝花': (26.5823, 101.7186), '雅安': (29.9818, 103.0133), '巴中': (31.8679, 106.7543),
        '资阳': (30.1286, 104.6270), '丽江': (26.8550, 100.2270), '大理': (25.6065, 100.2676),
        '玉溪': (24.3520, 102.5439), '保山': (25.1120, 99.1618), '昭通': (27.3382, 103.7175),
        '楚雄': (25.0453, 101.5460), '红河': (23.3631, 103.3750), '文山': (23.3692, 104.2443),
        '西双版纳': (22.0094, 100.7970), '德宏': (24.4334, 98.5844), '怒江': (25.8529, 98.8567),
        '迪庆': (27.8188, 99.7026), '海东': (36.5020, 102.1043), '海西': (37.3771, 97.3708),
        '海南州': (36.2866, 100.6195), '海北': (36.9544, 100.9009), '黄南': (35.5177, 102.0190),
        '果洛': (34.4714, 100.2447), '玉树': (32.9932, 97.0085), '林芝': (29.6547, 94.3613),
        '山南': (29.2370, 91.7731), '日喀则': (29.2670, 88.8806), '昌都': (31.1406, 97.1720),
        '那曲': (31.4807, 92.0578), '阿里': (32.5011, 80.1058), '铜川': (34.9089, 108.9440),
        '商洛': (33.8683, 109.9402), '安康': (32.6903, 109.0293), '汉中': (33.0675, 107.0233),
        '延安': (36.5853, 109.4898), '渭南': (34.4994, 109.5029), '宝鸡': (34.3619, 107.2373),
        '石嘴山': (38.9841, 106.3836), '吴忠': (37.9975, 106.1984), '固原': (36.0159, 106.2426),
        '中卫': (37.5149, 105.1968),
        # job_vec 中仍曾遗漏的地名（各 1 条样本亦收录，便于凑齐可落点条数）
        '吕梁': (37.5183, 111.1443), '常德': (29.0314, 111.6985), '益阳': (28.5539, 112.3552),
        '包头': (40.6574, 109.8403), '焦作': (35.2159, 113.2418), '商丘': (34.4143, 115.6564),
        '九江': (29.7051, 116.0019), '绵阳': (31.4677, 104.6790), '景德镇': (29.2689, 117.1784),
        '崇左': (22.4041, 107.3647), '枣庄': (34.8105, 117.3238), '济源': (35.0672, 112.6015),
        '鹰潭': (28.2602, 117.0694), '宜春': (27.8043, 114.3910), '茂名': (21.6627, 110.9254),
        '扬州': (32.3932, 119.4127), '秦皇岛': (39.9354, 119.6005), '衢州': (28.9417, 118.8743),
        '晋城': (35.4904, 112.8513), '新乡': (35.3030, 113.9268), '信阳': (32.1470, 114.0913),
        '陇南': (33.4010, 104.9214), '海宁': (30.5097, 120.6811), '毕节': (27.2985, 105.3055),
        '雄安新区': (38.9938, 115.9734), '怀化': (27.5695, 110.0040), '淮安': (33.6104, 119.0153),
        '呼伦贝尔': (49.2116, 119.7658), '遵义': (27.7257, 106.9274), '鹤壁': (35.7482, 114.2974),
        '周口': (33.6261, 114.6969), '铜仁': (27.7183, 109.1916),         '晋中': (37.6870, 112.7527), '黄冈': (30.4539, 114.8723),
        '吉林': (43.8434, 126.5496), '娄底': (27.7000, 111.9945), '阿拉善盟': (38.8512, 105.7289),
        '驻马店': (32.9773, 114.0250), '防城港': (21.6174, 108.3541), '汕尾': (22.7864, 115.3751),
        '大同': (40.0764, 113.3001), '鄂尔多斯': (39.6083, 109.7817), '武威': (37.9283, 102.6385),
    }


def _build_city_map_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'city' not in df.columns:
        return pd.DataFrame(columns=['city', 'count', 'lat', 'lon'])
    city_series = df['city'].dropna().astype(str).str.strip()
    if city_series.empty:
        return pd.DataFrame(columns=['city', 'count', 'lat', 'lon'])
    city_counts = city_series.apply(_normalize_city_name).value_counts().reset_index()
    city_counts.columns = ['city', 'count']
    coords = _city_coord_map()
    city_counts['lat'] = city_counts['city'].map(lambda c: coords[c][0] if c in coords else None)
    city_counts['lon'] = city_counts['city'].map(lambda c: coords[c][1] if c in coords else None)
    city_counts = city_counts.dropna(subset=['lat', 'lon']).copy()
    return city_counts


def _toolbar_panel() -> Optional[str]:
    """工具栏 Code/Data 与 Paper 一致：用 URL 查询参数切换面板（无 st.button 圆角框）。"""
    if 'panel' not in st.query_params:
        return None
    v = st.query_params['panel']
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if v in ('code', 'data'):
        return str(v)
    return None


# 地图同城多点随机偏移（单位：度）。纬度 1°≈111km；原 ±0.014≈±1.6km，增大后点更分散。
MAP_JITTER_LAT_DEG = 0.2
MAP_JITTER_LON_DEG = 0.2


def _jobs_geoplot_frame(df: pd.DataFrame) -> pd.DataFrame:
    """为每一行岗位生成 plot_lat/plot_lon：优先用表内 lat/lon；否则按 city 查表；固定种子随机抖动避免同城点重叠（幅度见 MAP_JITTER_*）。"""
    if df.empty:
        return pd.DataFrame()
    d = df.reset_index(drop=True)
    coords = _city_coord_map()

    if 'city' in d.columns:
        cn = d['city'].fillna('').astype(str).str.strip().apply(_normalize_city_name)
        clat = cn.map(lambda c: coords[c][0] if c in coords else np.nan).to_numpy(dtype=float)
        clon = cn.map(lambda c: coords[c][1] if c in coords else np.nan).to_numpy(dtype=float)
    else:
        clat = np.full(len(d), np.nan)
        clon = np.full(len(d), np.nan)

    if {'lat', 'lon'}.issubset(d.columns):
        nlat = pd.to_numeric(d['lat'], errors='coerce').to_numpy(dtype=float)
        nlon = pd.to_numeric(d['lon'], errors='coerce').to_numpy(dtype=float)
        has_native = np.isfinite(nlat) & np.isfinite(nlon)
        base_lat = np.where(has_native, nlat, clat)
        base_lon = np.where(has_native, nlon, clon)
    else:
        base_lat, base_lon = clat, clon

    ok = np.isfinite(base_lat) & np.isfinite(base_lon)
    if not ok.any():
        return pd.DataFrame()

    d = d.loc[ok].copy()
    base_lat = base_lat[ok]
    base_lon = base_lon[ok]
    rng = np.random.default_rng(42)
    m = len(d)
    d['plot_lat'] = base_lat + rng.uniform(-MAP_JITTER_LAT_DEG, MAP_JITTER_LAT_DEG, size=m)
    d['plot_lon'] = base_lon + rng.uniform(-MAP_JITTER_LON_DEG, MAP_JITTER_LON_DEG, size=m)
    return d


def _fmt_money(v: float) -> str:
    return f"¥{v:,.0f}"


def _report_paragraph(text: str) -> None:
    safe = html.escape((text or "").strip()).replace('\n', '<br>')
    st.markdown(f"<p class='report-paragraph'>{safe}</p>", unsafe_allow_html=True)


def _build_report_facts(jobs: pd.DataFrame) -> dict:
    parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    d = jobs.copy()
    if d.empty:
        return {}

    if 'avg_salary' in d.columns:
        sal = pd.to_numeric(d['avg_salary'], errors='coerce')
    elif 'salary_avg' in d.columns:
        sal = pd.to_numeric(d['salary_avg'], errors='coerce')
    else:
        sal = pd.Series(np.nan, index=d.index, dtype=float)
    d['_sal'] = sal

    if 'city' in d.columns:
        city = d['city'].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        city_norm = city.dropna().str.split('·').str[0].str.replace('市$', '', regex=True)
    else:
        city_norm = pd.Series(dtype=str)
    city_count = city_norm.value_counts().head(8)
    city_salary = (
        pd.DataFrame({'city': city_norm.values, 'salary': d.loc[city_norm.index, '_sal'].values})
        .dropna()
        .groupby('city')
        .agg(count=('salary', 'count'), avg_salary=('salary', 'mean'))
        .query('count >= 10')
        .sort_values('avg_salary', ascending=False)
        .head(5)
    )

    edu_col = 'education_required' if 'education_required' in d.columns else None
    if edu_col:
        edu = d[edu_col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        edu_stats = (
            d.assign(_edu=edu)
            .dropna(subset=['_edu', '_sal'])
            .groupby('_edu')
            .agg(count=('_sal', 'count'), avg_salary=('_sal', 'mean'))
            .sort_values('count', ascending=False)
        )
    else:
        edu_stats = pd.DataFrame(columns=['count', 'avg_salary'])

    ind_col = 'industry_group' if 'industry_group' in d.columns else ('industry' if 'industry' in d.columns else None)
    if ind_col:
        ind = d[ind_col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
        ind_stats = (
            d.assign(_ind=ind)
            .dropna(subset=['_ind', '_sal'])
            .groupby('_ind')
            .agg(count=('_sal', 'count'), avg_salary=('_sal', 'mean'))
            .sort_values('count', ascending=False)
        )
    else:
        ind_stats = pd.DataFrame(columns=['count', 'avg_salary'])

    cluster_dist = {}
    p_cluster = os.path.join(parent, 'clustered_output', 'cluster_profiles.csv')
    if os.path.exists(p_cluster):
        try:
            cp = pd.read_csv(p_cluster, low_memory=False)
            job_count = pd.to_numeric(cp.get('job_count'), errors='coerce')
            cluster_dist = {
                'total': int(len(cp)),
                'small_lt10': int((job_count < 10).sum()),
                'mid_10_69': int(((job_count >= 10) & (job_count < 70)).sum()),
                'large_ge70': int((job_count >= 70).sum()),
            }
        except Exception:
            cluster_dist = {}

    exp_growth_top = {}
    exp_cov = 0
    p_exp = os.path.join(parent, 'regression_output', 'exp_curve.csv')
    if os.path.exists(p_exp):
        try:
            exp = pd.read_csv(p_exp, low_memory=False)
            exp['years_experience'] = pd.to_numeric(exp.get('years_experience'), errors='coerce')
            exp['predicted_salary'] = pd.to_numeric(exp.get('predicted_salary'), errors='coerce')
            exp_cov = int(exp.get('industry_group', pd.Series(dtype=str)).astype(str).nunique())
            p0 = exp[exp['years_experience'] == 0].set_index('industry_group')['predicted_salary']
            p5 = exp[exp['years_experience'] == 5].set_index('industry_group')['predicted_salary']
            growth = ((p5 - p0) / p0).dropna().sort_values(ascending=False).head(3)
            exp_growth_top = {str(k): float(v) for k, v in growth.items()}
        except Exception:
            exp_cov = 0
            exp_growth_top = {}

    n_skill_impact = 0
    p_skill_impact = os.path.join(parent, 'regression_output', 'skill_impact.csv')
    if os.path.exists(p_skill_impact):
        try:
            si = pd.read_csv(p_skill_impact, low_memory=False)
            n_skill_impact = int(si.get('industry_group', pd.Series(dtype=str)).astype(str).nunique())
        except Exception:
            n_skill_impact = 0

    n_skill_robust = 0
    robust_top = pd.DataFrame(columns=['industry_group', 'skill', 'pure_skill_value'])
    p_skill_robust = os.path.join(parent, 'regression_output', 'skill_value_robust.csv')
    if os.path.exists(p_skill_robust):
        try:
            sv = pd.read_csv(p_skill_robust, low_memory=False)
            n_skill_robust = int(sv.get('industry_group', pd.Series(dtype=str)).astype(str).nunique())
            if {'industry_group', 'skill', 'pure_skill_value'}.issubset(sv.columns):
                robust_top = (
                    sv[['industry_group', 'skill', 'pure_skill_value']]
                    .assign(pure_skill_value=pd.to_numeric(sv['pure_skill_value'], errors='coerce'))
                    .dropna(subset=['pure_skill_value'])
                    .sort_values('pure_skill_value', ascending=False)
                    .head(3)
                )
        except Exception:
            n_skill_robust = 0

    return {
        'n_jobs': int(len(d)),
        'salary_mean': float(d['_sal'].mean()),
        'salary_median': float(d['_sal'].median()),
        'salary_q1': float(d['_sal'].quantile(0.25)),
        'salary_q3': float(d['_sal'].quantile(0.75)),
        'city_count': city_count,
        'city_salary': city_salary,
        'edu_stats': edu_stats,
        'ind_stats': ind_stats,
        'cluster_dist': cluster_dist,
        'exp_cov': exp_cov,
        'exp_growth_top': exp_growth_top,
        'n_skill_impact': n_skill_impact,
        'n_skill_robust': n_skill_robust,
        'robust_top': robust_top,
    }


def main():
    jobs = load_jobs()
    report_sections = load_report_sections()
    skill_merge_df = load_skill_merge_preview()
    industry_col = 'industry' if 'industry' in jobs.columns else 'industry_group' if 'industry_group' in jobs.columns else None

    # Top report-style centered header (title/subtitle/author) + three-icon toolbar
    pdf_path = _resolve_pdf_path()
    paper_href = _get_ready_pdf_href()

    report_title = f"{datetime.now().year}年春季国内就业市场洞察：<br>基于网络招聘数据的多维分析与建模"
    report_subtitle = '面向高校人才培养与行业供需的跨维度洞察与行动建议'
    report_author = 'CareerMind 就业研究院'

    # 主标题块自身的上下留白：改下一行 div 的 padding（如 24px 0）；整页主内容距浏览器顶栏另见顶部 <style> 里 .block-container
    st.markdown(f"""
    <div style='text-align:center;padding:18px 0;'>
      <h1 style='margin:0;font-weight:700;color:#ffffff'>{report_title}</h1>
      <h3 style='margin:6px 0 0 0;color:#cbd5e1'>{report_subtitle}</h3>
      <div style='margin-top:8px;'>
        <span style="
          font-size:1.05rem;
          font-weight:800;
          letter-spacing:0.06em;
          background:linear-gradient(90deg,#60a5fa 0%,#a78bfa 25%,#f472b6 50%,#f59e0b 75%,#22d3ee 100%);
          -webkit-background-clip:text;
          background-clip:text;
          color:transparent;
          text-shadow:0 0 14px rgba(96,165,250,0.28),0 0 26px rgba(244,114,182,0.22);
        ">{report_author}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    paper_icon = _find_icon_path('Paper.png')
    code_icon = _find_icon_path('code.png')
    data_icon = _find_icon_path('data.png') or _find_icon_path('data.svg')

    st.markdown(
        "<div class='careermind-toolbar-spacer' style='height:30px' aria-hidden='true'></div>",
        unsafe_allow_html=True,
    )
    # 三图标居中：外层留白列 + 中间三列
    _, toolbar_mid, _ = st.columns([1, 6, 1])
    with toolbar_mid:
        t1, t2, t3 = st.columns([1, 1, 1])
        with t1:
            paper_img_uri = _file_to_data_uri(paper_icon)
            href = paper_href
            if href:
                _render_paper_pdf_toolbar_component(href, paper_img_uri)
            if not href:
                if paper_img_uri:
                    st.markdown(
                        f"<div class='careermind-toolbar-icon' style='text-align:center;border:none;padding:0;margin:0'>"
                        f"<img src=\"{paper_img_uri}\" width=\"72\" alt=\"Paper\" style=\"opacity:0.45;border:none;\"/>"
                        f"<div style=\"color:#94a3b8;margin-top:6px;\">Paper</div></div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("<div style='text-align:center;color:#94a3b8'>Paper</div>", unsafe_allow_html=True)
                st.warning(
                    '报告 PDF 暂不可用；要启用请在系统上安装 WeasyPrint 及其依赖（libpango, libcairo 等）。'
                    '参考：https://doc.courtbouillon.org/weasyprint/stable/first_steps.html'
                )
        with t2:
            code_img_uri = _file_to_data_uri(code_icon)
            if code_img_uri:
                st.markdown(
                    f"<div class='careermind-toolbar-icon' style='text-align:center;border:none;padding:0;margin:0'>"
                    f"<a href=\"{CODE_REPO_URL}\" target=\"_blank\" rel=\"noopener noreferrer\" "
                    f"style=\"display:inline-block;text-decoration:none;color:#e6eef6;font-weight:600;\">"
                    f"<img src=\"{code_img_uri}\" width=\"72\" alt=\"\" style=\"display:block;margin:0 auto 6px;border:none;\"/>"
                    f"&lt;/Code&gt;</a></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='careermind-toolbar-icon' style='text-align:center'>"
                    f"<a href=\"{CODE_REPO_URL}\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"color:#e6eef6;font-weight:600;\">&lt;/Code&gt;</a></div>",
                    unsafe_allow_html=True,
                )
        with t3:
            data_img_uri = _file_to_data_uri(data_icon)
            if data_img_uri:
                st.markdown(
                    f"<div class='careermind-toolbar-icon' style='text-align:center;border:none;padding:0;margin:0'>"
                    f"<a href=\"{DATA_REPO_URL}\" target=\"_blank\" rel=\"noopener noreferrer\" "
                    f"style=\"display:inline-block;text-decoration:none;color:#e6eef6;font-weight:600;\">"
                    f"<img src=\"{data_img_uri}\" width=\"72\" alt=\"\" style=\"display:block;margin:0 auto 6px;border:none;\"/>"
                    f"Data</a></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='careermind-toolbar-icon' style='text-align:center'>"
                    f"<a href=\"{DATA_REPO_URL}\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"color:#e6eef6;font-weight:600;\">Data</a></div>",
                    unsafe_allow_html=True,
                )

    panel = _toolbar_panel()
    if panel == 'code':
        st.info('展示关键分析脚本（只读）。')
        if st.button('收起', key='toolbar_close_panel'):
            if 'panel' in st.query_params:
                del st.query_params['panel']
        code_files = ['etl.py', 'salary_regression.py', 'job_clustering.py', 'adapt_clustered_output.py']
        for cf in code_files:
            p = os.path.join(os.path.dirname(__file__), '..', cf)
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        txt = f.read()
                    st.subheader(cf)
                    st.code(txt[:4000], language='python')
                except Exception:
                    st.write(cf, '读取失败')
            else:
                st.write(cf, '— 文件未找到')
    elif panel == 'data':
        st.info('数据浏览与下载（来自 analysis/ 或 clustered_output/）。')
        if st.button('收起', key='toolbar_close_panel'):
            if 'panel' in st.query_params:
                del st.query_params['panel']
        parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        candidates = [
            os.path.join(parent, 'job_vec.csv'),
            os.path.join(parent, 'regression_output', 'skill_impact.csv'),
            os.path.join(parent, 'regression_output', 'exp_curve.csv'),
        ]
        for fpath in candidates:
            if os.path.exists(fpath):
                st.subheader(os.path.basename(fpath))
                try:
                    df_preview = pd.read_csv(fpath, nrows=100)
                    st.dataframe(df_preview)
                    csv_bytes = open(fpath, 'rb').read()
                    st.download_button('Download '+os.path.basename(fpath), data=csv_bytes, file_name=os.path.basename(fpath))
                except Exception:
                    st.write('无法预览', os.path.basename(fpath))
            else:
                st.write(os.path.basename(fpath) + ' — 未找到')

    # 四指标「上方」主区留白：紧挨在概览四 metric 之上（工具栏/Code/Data 展开区之后）。改 height 即可
    st.markdown(
        "<div class='careermind-toolbar-spacer' style='height:80px' aria-hidden='true'></div>",
        unsafe_allow_html=True,
    )

    # Sidebar controls（按你的需求：三个筛选条件 + 筛选结果展示）
    st.sidebar.header('筛选面板')
    query = st.sidebar.text_input('智能查询（模糊匹配，留空为不过滤）', value='学生会')

    industries = get_industries(jobs) if not jobs.empty else []
    penetration_industries = get_industries_from_cluster_profiles()
    default_industries = ['电子信息'] if '电子信息' in industries else (industries[:3] if industries else [])
    selected_industries = st.sidebar.multiselect('行业筛选', industries, default=default_industries)

    # salary slider (monthly, expects salary_avg in numeric RMB/month)
    min_salary = int(jobs['salary_avg'].min()) if ('salary_avg' in jobs and jobs['salary_avg'].dropna().any()) else 0
    max_salary = int(jobs['salary_avg'].max()) if ('salary_avg' in jobs and jobs['salary_avg'].dropna().any()) else 30000
    salary_min, salary_max = st.sidebar.slider('薪资阈值（RMB/月）', min_value=0, max_value=max(50000, max_salary), value=(min_salary, max_salary), step=500)

    map_use_filtered = st.sidebar.checkbox(
        '地图与筛选结果联动',
        value=False,
        help='开启：地图只显示当前筛选后的岗位；关闭：地图固定显示全部已载入岗位。',
    )

    # 读图补充说明功能保留，但不再放到交互控制台里
    use_llm_charts = False
    deepseek_key = _deepseek_api_key()

    # Filters
    df = jobs.copy()
    if selected_industries and industry_col:
        df = df[df[industry_col].isin(selected_industries)]
    if 'salary_avg' in df.columns:
        df = df[df['salary_avg'].between(salary_min, salary_max, inclusive='both')]

    if query:
        query = str(query)
        # 全字段模糊匹配：当前 df 的所有列都会参与 contains 匹配（任一列命中即保留）
        if not df.empty:
            mask = pd.Series(False, index=df.index)
            for c in df.columns:
                try:
                    mask = mask | df[c].astype(str).str.contains(query, case=False, na=False)
                except Exception:
                    # 极少数列类型异常时跳过该列，不影响整体筛选
                    continue
            df = df[mask]

    st.sidebar.markdown('---')
    st.sidebar.subheader('筛选结果')
    st.sidebar.metric('当前匹配岗位数', f'{len(df)}')
    if len(df) and 'salary_avg' in df.columns and df['salary_avg'].notna().any():
        st.sidebar.metric('筛选后平均薪资', format_currency(float(df['salary_avg'].mean())))
    if not df.empty:
        st.sidebar.caption('筛选结果（全部记录与全部字段）')
        hidden_cols = {'skills_list'}
        df_display = df.loc[:, [c for c in df.columns if c not in hidden_cols and not str(c).startswith('job_vec_')]]
        # 按「记录数 + 表头」动态高度，避免少量结果时出现大量空白行
        row_px = 35
        header_px = 38
        padding_px = 6
        dynamic_height = header_px + padding_px + row_px * len(df_display)
        st.sidebar.dataframe(df_display, use_container_width=True, height=dynamic_height)
    else:
        st.sidebar.caption('当前筛选无结果。')

    # (Header rendered above: report-style title/subtitle/author)

    overview = get_overview_stats(jobs)
    # 四指标占满主内容宽度（无外层 [1,5,1] 收窄）；列等宽、间距由 gap 统一控制
    c1, c2, c3, c4 = st.columns(4, gap='medium')
    c1.metric('岗位总数', f"{overview['total_jobs']}")
    c2.metric('行业数', f"{overview['industry_count']}")
    c3.metric('平均薪资（含缺省）', format_currency(overview['avg_salary']))
    top_inds = overview['top_industries'] if 'top_industries' in overview else None
    if top_inds is not None and not top_inds.empty:
        c4.metric('岗位最多行业', top_inds.iloc[0, 0])

    # 四指标「下方」与 Tab 导航之间的间距，改 height（px）即可
    st.markdown(
        "<div class='careermind-metrics-below-spacer' style='height:12px' aria-hidden='true'></div>",
        unsafe_allow_html=True,
    )

    tabs = st.tabs(['概览看板', '岗位地理分布', '学历与薪酬', '行业穿透', '报告'])

    # --- 概览看板 ---
    with tabs[0]:
        st.subheader('行业薪资与岗位分布')
        if not jobs.empty:
            group_col = industry_col or 'industry'
            count_col = 'job_id' if 'job_id' in jobs.columns else group_col
            grp = jobs.groupby(group_col).agg(count=(count_col, 'count'), avg_salary=('salary_avg', 'mean')).reset_index()
            grp = grp.sort_values('count', ascending=False)
            grp_top = grp.head(30)
            fig = px.bar(grp_top, x=group_col, y='avg_salary', color='avg_salary', title='行业平均薪资（RMB/月）', color_continuous_scale='magma')
            st.plotly_chart(fig, use_container_width=True)
            _render_chart_readme(
                '行业平均薪资（RMB/月）',
                chart_insights.insight_industry_salary_bar(grp_top, group_col),
                grp_top,
                'industry_salary_bar',
                use_llm_charts,
                deepseek_key,
            )

            fig2 = px.bar(grp_top, x=group_col, y='count', title='行业岗位数量')
            st.plotly_chart(fig2, use_container_width=True)
            _render_chart_readme(
                '行业岗位数量',
                chart_insights.insight_industry_count_bar(grp_top, group_col),
                grp_top,
                'industry_count_bar',
                use_llm_charts,
                deepseek_key,
            )
        else:
            st.info('未检测到岗位数据。请将处理好的 jobs_clean.csv 放入 career_mind_dashboard/data/')

    # --- 岗位地理分布（独立页，沿用侧栏筛选后的 df / 全量回退逻辑） ---
    with tabs[1]:
        st.subheader('岗位地理分布')
        map_source_df = df if map_use_filtered else jobs
        plot_df = _jobs_geoplot_frame(map_source_df)
        n_all = len(map_source_df)
        n_pts = len(plot_df)
        n_miss = n_all - n_pts
        if map_use_filtered and plot_df.empty and not jobs.empty:
            alt = _jobs_geoplot_frame(jobs)
            if not alt.empty:
                st.warning('当前筛选下没有可映射落点的岗位；下图已改为**全部已载入岗位**。若需只看筛选子集，请放宽条件。')
                map_source_df = jobs
                plot_df = alt
                n_all, n_pts, n_miss = len(map_source_df), len(plot_df), len(map_source_df) - len(plot_df)
        if plot_df.empty:
            st.info('未找到可映射的城市信息（请检查岗位表中 city 字段是否为空或均为未收录地名）。')
        else:
            src_note = '当前筛选后的岗位' if map_use_filtered else '**全部已载入岗位**（不受左侧行业/薪资/关键词筛选）'
            st.caption(
                f'地图数据源：{src_note}。共 **{n_all}** 条记录，可落点 **{n_pts}** 条（每条一个点；'
                f'优先用表内经纬度，否则按 city 映射；同城随机偏移约 ±{MAP_JITTER_LAT_DEG:.2f}°纬 / ±{MAP_JITTER_LON_DEG:.2f}°经，此参数可调）。'
                + (f' 另有 {n_miss} 条因 city 为空或地名未收录无法落点。' if n_miss else '')
            )
            hover_cols = [c for c in ('title', 'job_title', 'city', 'industry', 'salary_avg') if c in plot_df.columns]
            hover_name = 'title' if 'title' in plot_df.columns else ('job_title' if 'job_title' in plot_df.columns else None)
            fig_map = px.scatter_geo(
                plot_df,
                lat='plot_lat',
                lon='plot_lon',
                hover_name=hover_name,
                hover_data={c: True for c in hover_cols if c != hover_name},
                projection='natural earth',
                title='岗位地理分布（每点一条岗位）',
            )
            fig_map.update_traces(marker=dict(size=6, opacity=0.55, line=dict(width=0)))
            fig_map.update_layout(
                geo=dict(
                    scope='asia',
                    showland=True,
                    landcolor='rgb(28,35,48)',
                    bgcolor='rgba(0,0,0,0)',
                    showcountries=True,
                    countrycolor='rgb(70,80,90)',
                    # 初始视野：以中国中部为中心，经纬范围覆盖中国大陆与东亚近邻（略收窄全球 natural earth 的默认视野）
                    center=dict(lon=106, lat=34),
                    lonaxis=dict(range=[72, 136]),
                    lataxis=dict(range=[16, 50]),
                    projection=dict(type='natural earth', scale=1.35),
                ),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)
            _render_chart_readme(
                '岗位地理分布（每点一条岗位）',
                chart_insights.insight_geo_points(plot_df, map_source_df),
                plot_df,
                'geo_scatter_jobs',
                use_llm_charts,
                deepseek_key,
            )
            with st.expander('按城市汇总（岗位条数）', expanded=False):
                city_agg = _build_city_map_df(map_source_df)
                if not city_agg.empty:
                    st.dataframe(city_agg.sort_values('count', ascending=False), use_container_width=True)

    # --- 学历与薪酬（原「学历画像」） ---
    with tabs[2]:
        st.subheader('学历与薪酬')
        if not jobs.empty:
            edu = extract_education_labels(jobs)
            edu_df = pd.DataFrame({'education': edu, 'salary': jobs.get('salary_avg', pd.Series([pd.NA]*len(jobs)))})
            summary = edu_df.groupby('education').agg(count=('salary', 'count'), avg_salary=('salary', 'mean')).reset_index()
            summary = summary.sort_values('avg_salary', ascending=False, na_position='last').reset_index(drop=True)
            fig = px.bar(
                summary,
                x='education',
                y='avg_salary',
                title='学历与平均薪资（平均薪资从高到低）',
                color='avg_salary',
                color_continuous_scale='viridis',
                category_orders={'education': summary['education'].tolist()},
            )
            st.plotly_chart(fig, use_container_width=True)
            _render_chart_readme(
                '学历与平均薪资（平均薪资从高到低）',
                chart_insights.insight_education_salary(summary),
                summary,
                'edu_salary_bar',
                use_llm_charts,
                deepseek_key,
            )
            st.dataframe(summary)
        else:
            st.info('请提供岗位数据以生成学历与薪酬分析。')

    # --- 行业穿透 ---
    with tabs[3]:
        st.subheader('行业穿透分析')
        pen_opts = penetration_industries if penetration_industries else ['—']
        _pen_default = pen_opts.index('电子信息') if '电子信息' in pen_opts else 0
        industry_to_analyze = st.selectbox(
            '选择行业进行穿透分析',
            options=pen_opts,
            index=min(_pen_default, len(pen_opts) - 1),
            help='仅列出 cluster_profiles.csv 中已参与聚类/画像的行业组，按名称拼音排序。',
        )
        if industry_to_analyze and industry_to_analyze != '—':
            clusters = get_clusters_for_industry(industry_to_analyze)
            if clusters.empty:
                st.info(
                    '未找到该行业在 cluster_profiles 中的聚类行（行业名需与聚类结果表完全一致）。'
                    '若岗位数据中有「消费品」等未参与聚类的行业，将不会出现穿透结果。'
                )
            else:
                st.write('聚类概览（仅当前行业）')
                st.dataframe(clusters, use_container_width=True)

                for idx, row in clusters.reset_index(drop=True).iterrows():
                    st.markdown(
                        f"<div class='card'><b>Cluster {row.get('cluster_id', idx)}</b> — 样本量: {row.get('sample_size', row.get('count', '—'))}</div>",
                        unsafe_allow_html=True,
                    )
                    core = row.get('core_skills', '') or ''
                    salary = row.get('salary_min_avg', row.get('salary_max_avg', None))
                    exp_txt = row.get('experience', '') or row.get('experience_required', '')
                    oth_txt = row.get('other_requirements', '') or row.get('other_requirement', '')
                    cols = st.columns([1, 2.2], vertical_alignment='top')
                    cols[0].metric('Average Salary', format_currency(salary))
                    with cols[1]:
                        st.markdown('<p class="cluster-field-label" style="margin-top:0;">核心技能</p>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="cluster-core-skills">{html.escape(str(core))}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('<p class="cluster-field-label">经验要求</p>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="cluster-field-body">{html.escape(str(exp_txt) if str(exp_txt).strip() else "—")}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('<p class="cluster-field-label">其他要求</p>', unsafe_allow_html=True)
                        st.markdown(
                            f'<div class="cluster-field-body">{html.escape(str(oth_txt) if str(oth_txt).strip() else "—")}</div>',
                            unsafe_allow_html=True,
                        )
                    sample_size = int(row.get('sample_size') or row.get('count') or 0)
                    if sample_size < 10:
                        st.markdown(
                            '<div class="cluster-sample-hint">'
                            + html.escape('该聚类样本量较小；下方「行业建模参考」仍为行业整体回归/稳健结果，供对照。')
                            + '</div>',
                            unsafe_allow_html=True,
                        )
                    elif sample_size <= 70:
                        st.markdown(
                            '<div class="cluster-sample-hint">'
                            + html.escape('该聚类为中等样本；技能与经验曲线见下方行业级图表。')
                            + '</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown('---')
                st.subheader(f'「{industry_to_analyze}」行业建模参考')
                st.caption(
                    '技能与经验–薪资曲线来自 `skill_impact.csv` / `skill_value_robust.csv` 与 `exp_curve.csv`，'
                    '均为**当前所选行业**整体估计（与单个聚类 ID 无关）；未纳入回归的行业将使用示意曲线。'
                )
                skill_imp = get_skill_importance(None, industry=industry_to_analyze)
                if not skill_imp.empty:
                    fig_sk = px.bar(
                        skill_imp.sort_values('importance', ascending=False).head(20),
                        x='skill',
                        y='importance',
                        title='技能重要性（按 |系数| 或稳健溢价归一化，取前 20）',
                    )
                    st.plotly_chart(fig_sk, use_container_width=True)
                    _render_chart_readme(
                        f'技能重要性（行业整体）— {industry_to_analyze}',
                        chart_insights.insight_skill_importance(skill_imp, coarse=False),
                        skill_imp.head(25),
                        f'industry_skill_{chart_insights.digest_for_df(skill_imp, industry_to_analyze)}',
                        use_llm_charts,
                        deepseek_key,
                    )
                else:
                    st.warning('未找到该行业的技能回归输出（skill_impact / skill_value_robust）。')

                exp = get_exp_curve(None, industry=industry_to_analyze)
                if not exp.empty:
                    exp_plot = exp.copy()
                    if 'exp_years' not in exp_plot.columns:
                        if 'years_experience' in exp_plot.columns:
                            exp_plot['exp_years'] = pd.to_numeric(exp_plot['years_experience'], errors='coerce')
                        elif 'year' in exp_plot.columns:
                            exp_plot['exp_years'] = pd.to_numeric(exp_plot['year'], errors='coerce') - 2023
                        else:
                            exp_plot['exp_years'] = np.arange(len(exp_plot), dtype=float)
                    exp_plot = exp_plot.dropna(subset=['exp_years', 'salary']).sort_values('exp_years')
                    fig_ex = px.line(exp_plot, x='exp_years', y='salary', title='经验–薪资参考曲线（行业整体，经验年数 0–10）')
                    fig_ex.update_xaxes(
                        title='经验年数',
                        tickmode='linear',
                        dtick=1,
                        tick0=0,
                        range=[0, 10.1],
                    )
                    st.plotly_chart(fig_ex, use_container_width=True)
                    _render_chart_readme(
                        f'经验–薪资参考曲线（行业整体，经验年数）— {industry_to_analyze}',
                        chart_insights.insight_exp_salary_curve(exp_plot),
                        exp_plot,
                        f'industry_exp_{chart_insights.digest_for_df(exp_plot, industry_to_analyze)}',
                        use_llm_charts,
                        deepseek_key,
                    )

    # --- 报告闭环 ---
    with tabs[4]:
        st.subheader('研究报告')
        _, report_col, _ = st.columns([1.2, 7.6, 1.2])
        with report_col:
            facts = _build_report_facts(jobs)

            overview_text = (
                "本报告基于 CareerMind 数据平台在特定采集周期内的招聘信息样本，旨在从显性岗位需求和技能画像角度，分析互联网行业及相关用人市场的结构性特征。"
                "由于数据来源于公开招聘渠道，本文采用定量分析与文本挖掘相结合的方法，探索院校、技能、地域等核心维度的聚集与差异。"
                "需要强调的是，本研究侧重于“可见的”能力要求与岗位标签，力求在现有数据框架下提供稳健的行业洞察，而不试图覆盖所有隐性就业机制。"
                "本文结论主要反映平台内招聘生态，对于外部渠道或非公开招聘行为的解释仍需谨慎。"
            )
            intro_text = (
                "在经济结构持续转型与产业升级加速推进的背景下，就业市场正呈现显著的动态性与结构性变化。"
                "传统就业分析较多依赖宏观统计或问卷数据，在时效性、颗粒度和可复核性方面存在局限。"
                "随着招聘平台数据可得性提高，岗位文本、薪资区间、学历门槛、经验要求和行业属性为微观就业结构刻画提供了新的证据基础。"
                "基于此，本项目构建“聚类画像 + 行业建模”的双层框架：前者刻画行业内部岗位类型及技能组合，后者量化经验与技能对薪资的作用。"
                "研究目标是从数据驱动角度识别结构性机会与约束，为高校培养方案、求职路径设计与企业岗位策略提供可执行参考。"
            )
            st.markdown('**概述**')
            _report_paragraph(overview_text)
            st.markdown('**引言**')
            _report_paragraph(intro_text)

            if facts:
                top_cities_text = '、'.join([f"{c}（{int(n)}）" for c, n in facts['city_count'].head(6).items()]) or '—'
                city_salary_top = '、'.join(
                    [f"{idx}（{_fmt_money(row['avg_salary'])}）" for idx, row in facts['city_salary'].head(4).iterrows()]
                ) or '—'
                st.markdown('**地域视角就业情况分析**')
                _report_paragraph(
                    f"当前样本共 {facts['n_jobs']} 条岗位，平均薪资 {_fmt_money(facts['salary_mean'])}，中位数 {_fmt_money(facts['salary_median'])}，"
                    f"IQR 区间为 {_fmt_money(facts['salary_q1'])}–{_fmt_money(facts['salary_q3'])}。岗位分布呈现明显城市集聚，"
                    f"岗位量前列城市为：{top_cities_text}。在样本量不少于 10 的城市中，平均薪资相对领先的城市包括：{city_salary_top}。"
                    "这说明一线与强二线城市仍是高质量岗位的主要承载区，同时岗位规模与薪资水平并不总是同步，"
                    "求职决策需要同时关注“机会密度”和“薪资水平”两类信号。"
                )

                edu = facts['edu_stats']
                edu_count_top = '、'.join(
                    [f"{idx}（{int(row['count'])}）" for idx, row in edu.head(4).iterrows()]
                ) if not edu.empty else '—'
                edu_salary_top = '、'.join(
                    [f"{idx}（{_fmt_money(row['avg_salary'])}）" for idx, row in edu.sort_values('avg_salary', ascending=False).head(4).iterrows()]
                ) if not edu.empty else '—'
                st.markdown('**学历视角就业分析**')
                _report_paragraph(
                    f"学历结构上，岗位需求主要集中在：{edu_count_top}。从薪资均值看，领先学历层次为：{edu_salary_top}。"
                    "整体上，学历要求与薪资水平呈正相关，但并非线性单调：应用型岗位在“大专/本科”层级形成了更大的岗位池，"
                    "而硕博岗位虽薪资更高，但样本规模较小、赛道集中。对高校而言，课程设计应在“学历门槛”之外，"
                    "更强调可迁移能力与岗位技能栈的可验证性。"
                )

                ind = facts['ind_stats']
                ind_count_top = '、'.join(
                    [f"{idx}（{int(row['count'])}）" for idx, row in ind.head(5).iterrows()]
                ) if not ind.empty else '—'
                ind_salary_top_df = ind[ind['count'] >= 50].sort_values('avg_salary', ascending=False).head(5)
                ind_salary_top = '、'.join(
                    [f"{idx}（{_fmt_money(row['avg_salary'])}）" for idx, row in ind_salary_top_df.iterrows()]
                ) if not ind_salary_top_df.empty else '—'
                cluster_dist = facts['cluster_dist']
                exp_growth = '、'.join(
                    [f"{k}（0-5年约+{v*100:.1f}%）" for k, v in facts['exp_growth_top'].items()]
                ) or '—'
                robust_top = '、'.join(
                    [
                        f"{r['industry_group']}-{r['skill']}（约{_fmt_money(float(r['pure_skill_value']))}）"
                        for _, r in facts['robust_top'].iterrows()
                    ]
                ) or '—'
                st.markdown('**行业视角就业分析**')
                _report_paragraph(
                    f"按岗位规模看，当前样本需求最集中的行业包括：{ind_count_top}。在样本量不少于 50 的行业中，"
                    f"平均薪资领先行业为：{ind_salary_top}。这说明当前劳动力需求既呈现“制造/工程等大体量行业吸纳岗位”，"
                    "也呈现“数字技术与高壁垒行业拉高薪资中枢”的双轨特征。"
                )
                _report_paragraph(
                    f"聚类画像方面，已形成 {cluster_dist.get('total', 0)} 个画像，其中小样本(<10) {cluster_dist.get('small_lt10', 0)} 个、"
                    f"中样本(10-69) {cluster_dist.get('mid_10_69', 0)} 个、大样本(>=70) {cluster_dist.get('large_ge70', 0)} 个。"
                    "聚类结果有效揭示了“同一行业内部”的岗位异质性：即便行业标签一致，不同岗位簇在核心技能、经验要求与薪资区间上仍存在显著分化。"
                    "因此在解读行业结论时，不能只看行业均值，还应结合簇级画像识别真正的高潜岗位群。"
                )
                _report_paragraph(
                    f"薪资建模方面，经验曲线覆盖 {facts['exp_cov']} 个行业，0-5 年薪资增速较快行业为：{exp_growth}。"
                    f"技能价值层面，大样本回归覆盖 {facts['n_skill_impact']} 个行业，中小样本稳健估计覆盖 {facts['n_skill_robust']} 个行业；"
                    f"稳健溢价示例包括：{robust_top}。这组结果表明，行业薪资差距并非仅由经验年限决定，技能组合与行业情境的交互作用同样关键。"
                )

                st.markdown('**深度讨论**')
                _report_paragraph(
                    f"{report_sections.get('discussion', '').strip()} "
                    "结合本次数据可见，行业间差异并不只体现在“起薪高低”，更体现在“成长斜率”和“技能回报结构”上。"
                    "对高校与培训体系来说，教学供给需要从“通用能力堆叠”转向“行业场景能力包”；"
                    "对求职者来说，应以目标行业的高回报技能簇为核心构建学习路径，而非在跨行业技能上平均用力。"
                )

                st.markdown('**反思与局限**')
                _report_paragraph(
                    f"{report_sections.get('limitations', '').strip()} "
                    "进一步看，本研究虽通过聚类与回归提升了结构解释力，但仍基于公开招聘文本，难以覆盖企业内推、校友网络与岗位隐性门槛等非公开机制。"
                    "这意味着我们的结论更适用于“显性需求市场”，而不代表全部就业通道。"
                )
                _report_paragraph(
                    f"在当前样本中，小样本聚类簇仍有 {cluster_dist.get('small_lt10', 0)} 个，相关画像更适合作为定性线索而非稳健统计结论；"
                    "城市字段也存在“城市-区县”混合粒度，可能放大地域对比的口径差异。部分技能溢价极值还可能受到样本稀疏、岗位标题噪声与行业内岗位异质性的共同影响。"
                    "后续可通过多平台融合、时间滚动验证与更严格的因果识别策略，提升外推性与稳健性。"
                )

                st.markdown('**结论**')
                _report_paragraph(
                    "综合来看，本期就业市场呈现“城市集聚 + 行业分化 + 学历分层 + 技能异质回报”并存格局。"
                    "求职侧建议围绕目标行业构建技能组合并关注区域匹配；高校侧建议强化与产业链协同的课程模块与项目实践；"
                    "用人侧建议在岗位描述中提高能力标签的可验证性和透明度，以降低匹配摩擦并提升招聘效率。"
                )
            else:
                st.warning('当前未加载到可用于报告写作的数据，无法生成数据驱动章节。')

            st.markdown('**参考文献**')
            _report_paragraph("1) OECD. Education at a Glance 2025。")
            _report_paragraph("2) World Bank. World Development Report 2026。")
            _report_paragraph("3) Autor, D. Skills, Education, and the Rise of Earnings Inequality。")
            _report_paragraph("4) Card, D. The Causal Effect of Education on Earnings。")
            _report_paragraph("5) 国内招聘平台公开岗位数据与 CareerMind 项目内部 ETL/建模文档。")

            st.markdown('**附录：数据概况与方法论**')
            _report_paragraph(
                "本项目流程为：jobs().csv 原始抓取 -> temp.py 从 description 提取 other_requirement 得到 jobs(1).csv -> "
                "etl.py 完成经验数字化、技能共线合并、薪资统一与向量化得到 job_vec.csv，并输出 skill_merge_preview.csv。"
                "随后 job_clustering.py 分行业生成聚类画像（clustered_output/cluster_profiles.csv 与分行业文件），"
                "salary_regression.py 分行业生成经验曲线与技能价值估计（regression_output/exp_curve.csv、skill_impact.csv、skill_value_robust.csv）。"
                "网页端报告使用上述产物进行可视化与文字分析。详情见页首 Code 或 Data 链接。"
            )

            st.markdown('**附录：技能标签合并对应关系**')
            _report_paragraph(report_sections.get('appendix_skill_merge_intro', '—'))
            if not skill_merge_df.empty:
                st.dataframe(skill_merge_df, use_container_width=True, height=420)
            else:
                st.caption('未加载到 skill_merge_preview.csv；可在项目根运行 python data_transfer_to_dashboard.py 同步到 data/。')

    st.markdown('---')
    st.caption('数据来源：51job。报告由 CareerMind 数据平台分析并撰写。')


if __name__ == '__main__':
    main()
