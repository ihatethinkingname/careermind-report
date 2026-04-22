import os
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
INPUT_CSV = os.path.join(DATA_DIR, "jobs().csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "jobs(1).csv")
MODEL_NAME = "local_model/paraphrase-multilingual-MiniLM-L12-v2"

# Seed phrases for semantic classification
SKILL_PROMPTS = [
    "熟悉Python、SQL、Excel等数据分析工具",
    "掌握PLC编程与电气自动化设备调试",
    "具备项目管理、需求分析和实施能力",
    "熟练使用Office办公软件和项目管理工具",
    "熟悉软件开发流程、数据库和系统集成",
]

EXPERIENCE_PROMPTS = [
    "至少3年相关工作经验",
    "5年以上制造业设备维护管理经验",
    "无需经验，实习生也可",
    "有项目管理经验者优先",
    "具备2年以上团队管理经验",
]

OTHER_PROMPTS = [
    "具备良好的沟通协调能力和团队合作精神",
    "责任心强、抗压能力强、学习能力强",
    "能够适应高强度工作节奏，服从安排",
    "具有良好的职业素养和服务意识",
    "愿意出差，能接受不定期加班",
]

IGNORE_PATTERNS = [
    r'五险一金|带薪年假|年终奖金|绩效奖金|岗位福利|福利待遇|补贴|保险|住宿|餐补',
    r'薪资\s*\d+|月薪|年薪|工资|待遇',
    r'工作时间|周末双休|加班|9:00|18:00|午休|排班|休假',
    r'公司介绍|企业文化|团队年轻|氛围.*好|氛围轻松|期待你|等你来聊|加入我们',
    r'职位发布|邀约|面试|入职跟踪|招聘全流程',
    r'产品.*线|客户.*沟通|合作伙伴|业务拓展',
    r'办公地点|工作地点|联系方式|联系电话|邮箱',
    r'HRBP|船东|公司提供食宿|船舶|杂货船|周期|个月左右|\d+个月|本岗位|本公司|招聘人数|\d+\s*人|岗位性质|岗位类别',
    r'年龄要求|职能类别|学历要求|工作内容|岗位职责',
]

SECTION_HEADER_PATTERN = r'(任职要求|岗位要求|任职资格|任职条件)'
SECTION_END_PATTERN = r'(年龄要求|职能类别|工作内容|岗位职责|职位详情|福利待遇|薪资福利|联系我们|公司介绍|公司福利|福利|岗位职责)'


def load_model():
    try:
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception as exc:
        print(
            "Warning: unable to load the local sentence-transformers model. "
            "Falling back to regex-based classification."
        )
        print("Model load error:", exc)
        return None

model = load_model()
SKILL_EMB = None
EXP_EMB = None
OTHER_EMB = None
if model is not None:
    SKILL_EMB = model.encode(SKILL_PROMPTS, convert_to_tensor=True)
    EXP_EMB = model.encode(EXPERIENCE_PROMPTS, convert_to_tensor=True)
    OTHER_EMB = model.encode(OTHER_PROMPTS, convert_to_tensor=True)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return text.replace("\r", "\n").replace("\u3000", " ").strip()


def clean_line(line):
    if not isinstance(line, str):
        return ""
    text = line.strip()
    text = re.sub(r'^[\-–—·•\s【】\[\]]+', '', text)
    text = re.sub(r'[\s\-–—·•【】\[\]]+$', '', text)
    text = re.sub(r'^[：:]+', '', text)
    return text.strip()


def split_text_into_lines(text):
    text = normalize_text(text)
    if not text:
        return []
    text = re.sub(r'([。；;])', r'\1\n', text)
    text = re.sub(r'(?<!\d)([：:])', r'\1\n', text)
    parts = [clean_line(item) for item in text.splitlines()]
    return [part for part in parts if part and not re.match(r'^(职位详情|工作内容|任职要求|岗位要求|任职资格|年龄要求|职能类别)[:：]?$', part)]


def extract_requirements_section(description):
    text = normalize_text(description)
    header_match = re.search(SECTION_HEADER_PATTERN, text)
    if not header_match:
        # fallback to a line-start 要求: pattern, avoid matching sentence-internal 要求
        header_match = re.search(r'(^|\n)\s*要求[:：]?', text)
    if not header_match:
        return text
    start = header_match.end()
    tail = text[start:]
    end_match = re.search(SECTION_END_PATTERN, tail)
    if end_match:
        return tail[:end_match.start()]
    return tail


def is_ignore_line(line):
    if not line:
        return True
    for pattern in IGNORE_PATTERNS:
        if re.search(pattern, line, re.I):
            return True
    return False


def classify_line(line):
    if not line:
        return "ignore"
    line = clean_line(line)
    if not line:
        return "ignore"

    if is_ignore_line(line):
        return "ignore"

    if re.search(r'学历|大专|本科|高中|硕士|博士|学位|应届毕业生|毕业生|专业背景|专业优先', line):
        return "other"

    if re.search(r'年龄|岁', line) and not re.search(r'经验|年', line):
        return "other"

    if re.search(r'无需经验|经验者优先|从业经验|工作经验|\d+\s*年|年以上|年及以上|半年|实习|经验丰富', line):
        return "experience"

    if re.search(r'熟悉|熟练|掌握|了解|能够|擅长|技能|工具|软件|系统|平台|编程|开发|测试|运维|设计|分析|优化|实施|配置|数据库|项目管理|沟通协调|沟通能力|团队协作|团队合作|写作|表达|英语|PMP|CISP|SAP|Oracle|ERP|电脑|office|文档|报告|操作|维护|压力|工作压力', line):
        return "skill"

    if model is None or SKILL_EMB is None or EXP_EMB is None or OTHER_EMB is None:
        return "other"

    emb = model.encode(line, convert_to_tensor=True)
    sim_skill = util.cos_sim(emb, SKILL_EMB).max().item()
    sim_exp = util.cos_sim(emb, EXP_EMB).max().item()
    sim_other = util.cos_sim(emb, OTHER_EMB).max().item()

    if sim_exp >= sim_skill and sim_exp >= sim_other and sim_exp > 0.45:
        return "experience"
    if sim_skill >= sim_other and sim_skill > 0.45:
        return "skill"
    if sim_other > 0.40:
        return "other"
    if sim_skill > sim_exp:
        return "skill"
    if sim_exp > sim_skill:
        return "experience"
    return "other"


def extract_other_requirement(description):
    section = extract_requirements_section(description)
    lines = split_text_into_lines(section)
    other_lines = []
    for line in lines:
        if not line:
            continue
        segments = [clean_line(seg) for seg in re.split(r'[；;，,]+', line) if clean_line(seg)]
        for seg in segments:
            label = classify_line(seg)
            if label == "other":
                other_lines.append(seg)
    return "；".join(other_lines)


def build_other_requirement_column(df):
    df = df.copy()
    df["other_requirement"] = df["job_description"].fillna("").apply(extract_other_requirement)
    return df


def main():
    df = pd.read_csv(INPUT_CSV, dtype=str)
    if "other_requirement" not in df.columns:
        df = build_other_requirement_column(df)
    else:
        df["other_requirement"] = df["job_description"].fillna("").apply(extract_other_requirement)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved {OUTPUT_CSV} with other_requirement column.")


if __name__ == "__main__":
    main()
