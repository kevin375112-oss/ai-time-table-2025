# app.py — Streamlit Cloud 최종 완벽 버전
import streamlit as st
import pandas as pd
import os
import re
import random
from sentence_transformers import SentenceTransformer, util

# ==================== 1. CSV 강제 로드 (Streamlit Cloud 필수!) ====================
for i in range(1, 8):
    src = f"/mount/src/ai-time-table-2025/section{i}.csv"
    dst = f"section{i}.csv"
    if os.path.exists(src) and not os.path.exists(dst):
        os.system(f"cp {src} {dst}")

# ==================== 2. 기본 설정 ====================
st.set_page_config(page_title="2025 AI 시간표 생성기", layout="wide")
st.title("2025-2학기 AI 시간표 생성기")
st.markdown("**전공 고정 │ 시간 겹침 0% │ '운동' '영어' '경제'만 써도 AI가 알아서 추천**")

# ==================== 3. 전공 고정 시간표 ====================
FIXED_SCHEDULE = [
    {"name": "공학수학", "time": "화 9:00-10:15, 목 9:00-10:15", "prof": "강수진"},
    {"name": "고전읽기와토론", "time": "월 9:00-10:40", "prof": "황미은"},
    {"name": "일반화학2", "time": "월 15:00-16:15, 수 15:00-16:15", "prof": "조혜진"},
    {"name": "인공지능프로그래밍", "time": "화 13:30-14:45, 목 13:30-14:45", "prof": "이휘돈"},
    {"name": "일반물리학2", "time": "화 16:30-17:45, 목 16:30-17:45", "prof": "양하늬"},
]
AREAS = {1:"사상/역사", 2:"사회/문화", 3:"문학/예술", 4:"과학/기술", 5:"건강/레포츠", 6:"외국어", 7:"융복합"}

# ==================== 4. AI 모델 로딩 ====================
@st.cache_resource
def load_model():
    with st.spinner("AI 모델 로딩 중… (최초 30초 정도)"):
        return SentenceTransformer('jhgan/ko-sroberta-multitask')
model = load_model()

# ==================== 5. 강력한 시간 파싱 (368개 전부 잡음) ====================
def parse_time(text):
    if not isinstance(text, str): return []
    text = re.sub(r'<br\s*/?>|\n|,', ' ', text)
    slots = []
    yoil_map = {"월":0, "화":1, "수":2, "목":3, "금":4}
    patterns = [
        r"([월화수목금])\s*(\d{1,2}:\d{2})\s*[-~]\s*(\d{1,2}:\d{2})",
        r"([월화수목금])\s*(\d{1,2}:\d{2})\s*\(\s*(\d+)\s*\)"
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            day = yoil_map.get(m.group(1))
            if not day: continue
            start = sum(int(x)*60**i for i,x in enumerate(reversed(m.group(2).split(":"))))
            if len(m.groups()) == 3 and m.group(3).isdigit():
                end = start + int(m.group(3))
            else:
                end = sum(int(x)*6060**i for i,x in enumerate(reversed(m.group(3).split(":"))))
            if end > start:
                slots.append({"day": day, "start": start, "end": end})
    return slots

# ==================== 6. 데이터 로드 ====================
fixed_courses = []
for i, d in enumerate(FIXED_SCHEDULE):
    slots = parse_time(d["time"])
    if slots:
        fixed_courses.append({**d, "id": f"maj_{i}", "type": "major", "slots": slots})

courses = []
for i in range(1, 8):
    if os.path.exists(f"section{i}.csv"):
        df = pd.read_csv(f"section{i}.csv", encoding="cp949").fillna("")
        for _, r in df.iterrows():
            name = str(r.get("교과목명(미확정구분)", "")).strip()
            prof = str(r.get("교수명", "미정")).strip()
            time_str = str(r.get("시간/강의실", ""))
            slots = parse_time(time_str)
            if name and slots:
                courses.append({
                    "id": len(courses), "name": name, "prof": prof,
                    "area": i, "type": "general", "slots": slots,
                    "search_text": f"{name} {AREAS[i]} {prof}"
                })

if not courses:
    st.error("CSV 파일을 읽지 못했습니다!")
    st.stop()

st.success(f"총 {len(courses)}개 교양 과목 로드 완료!")

# 임베딩 미리 계산
course_embeddings = model.encode([c["search_text"] for c in courses], convert_to_tensor=True)

# ==================== 7. AI 엔진 ====================
def calculate_scores(keyword):
    for c in courses: c["match_score"] = 0.0
    if not keyword: return
    q = model.encode(keyword, convert_to_tensor=True)
    scores = util.cos_sim(q, course_embeddings)[0].cpu().numpy()
    for i, s in enumerate(scores):
        courses[i]["match_score"] = float(s) * 100
    for c in courses:
        if keyword in c["name"]:
            c["match_score"] += 50

def check_overlap(sched):
    slots = sorted([(s["day"], s["start"], s["end"]) for c in sched for s in c["slots"]])
    return any(i < len(slots)-1 and slots[i][0]==slots[i+1][0] and slots[i][2] > slots[i+1][1] 
               for i in range(len(slots)-1))

def run_ai(target_areas, pick_n, keyword=""):
    calculate_scores(keyword)
    pool = [c for c in courses if c["area"] in target_areas]
    if keyword:
        pool = [c for c in pool if c["match_score"] > 30]
    pool.sort(key=lambda x: -(x["match_score"]*5 + random.random()))
    pool = pool[:60]

    results = []
    for _ in range(2500):
        picks = random.sample(pool, min(len(pool), pick_n))
        schedule = fixed_courses + picks
        if not check_overlap(schedule):
            score = sum(c["match_score"] for c in picks)
            results.append({"score": score, "schedule": schedule})
            if len(results) >= 5: break
    return sorted(results, key=lambda x: -x["score"])[:3]

# ==================== 8. 시간표 HTML 렌더링 ====================
def render_timetable(schedule):
    PX = 1.35
    H_START, H_END = 9, 19
    html = f"""
    <style>
        .tt{{display:flex;font-family:Malgun Gothic;font-size:13px;border:1px solid #ddd;width:100%}}
        .tc{{position:relative;border-right:1px solid #eee;height:{(H_END-H_START)*60*PX}px;flex:1}}
        .card{{position:absolute;width:93%;left:3.5%;padding:6px;border-radius:6px;
               box-shadow:2px 2px 8px rgba(0,0,0,0.15);text-align:center}}
    </style>
    <div style="display:flex;margin-left:60px">
        {''.join(f'<div style="flex:1;text-align:center;padding:8px;background:#333;color:white;font-weight:bold">{d}</div>' for d in "월화수목금")}
    </div>
    <div class="tt"><div style="width:60px;background:#fafafa;position:relative">
        {''.join(f'<div style="position:absolute;top:{(h-H_START)*60*PX}px;width:100%;text-align:right;padding-right:8px;font-size:11px;color:#666">{h:02d}:00</div>' for h in range(H_START, H_END+1))}
    </div>"""
    for day in range(5):
        html += "<div class='tc'>"
        for c in schedule:
            for s in c["slots"]:
                if s["day"] == day:
                    top = (s["start"] - H_START*60) * PX
                    hgt = (s["end"] - s["start"]) * PX
                    if c["type"] == "major":
                        bg, color, tag = "#e3f2fd", "#1976d2", "전공"
                    elif c.get("match_score",0) > 80:
                        bg, color, tag = "#ffebee", "#d32f2f", "강력추천"
                    elif c.get("match_score",0) > 40:
                        bg, color, tag = "#e8f5e9", "#388e3c", "AI추천"
                    else:
                        bg, color, tag = "#fff3e0", "#f57c00", AREAS.get(c["area"],"교양")
                    html += f'<div class="card" style="top:{top}px;height:{hgt}px;background:{bg};border-left:5px solid {color};color:{color}">'
                    html += f"<small>{tag}</small><br><b>{c['name']}</b><br>{c['prof']}</div>"
        html += "</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ==================== 9. UI ====================
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("영역 선택")
    selected = []
    for i, name in AREAS.items():
        if st.checkbox(name, key=f"a{i}"):
            selected.append(i)

with col2:
    st.subheader("설정")
    num = st.selectbox("교양 과목 수", [1, 2, 3], index=1)
    keyword = st.text_input("AI 키워드 검색", placeholder="예: 운동, 영어, 경제, 철학, 코딩")

if st.button("시간표 생성", type="primary"):
    if not selected:
        st.error("영역을 하나 이상 선택해주세요!")
    else:
        with st.spinner("AI가 최고의 시간표를 찾는 중…"):
            results = run_ai(selected, num, keyword)
        if not results:
            st.error("조건에 맞는 시간표를 찾지 못했습니다.")
        else:
            for idx, r in enumerate(results):
                score = r["score"]
                with st.expander(f"추천 {idx+1}위 (AI 점수: {score:.1f})"):
                    for c in r["schedule"]:
                        if c["type"] == "general":
                            tag = "강력추천" if c.get("match_score",0) > 80 else ("AI추천" if c.get("match_score",0) > 40 else "")
                            st.write(f"• {c['name']} ({c['prof']}) {tag}")
                    render_timetable(r["schedule"])
