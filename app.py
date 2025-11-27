# app.py — 진짜 진짜 진짜 최종 완성본 (2025.11.27 새벽)
import streamlit as st
import pandas as pd
import os
import re
import random
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# ===================== CSV 강제 로드 (Streamlit Cloud 필수) =====================
for i in range(1, 8):
    src = f"/mount/src/ai-time-table-2025/section{i}.csv"
    dst = f"section{i}.csv"
    if os.path.exists(src) and not os.path.exists(dst):
        os.system(f"cp {src} {dst}")

st.set_page_config(page_title="2025 AI 시간표 생성기", layout="wide")
st.title("2025-2학기 AI 시간표 생성기")
st.markdown("**전공 고정 │ 시간 겹침 0% │ '운동' '영어' '코딩'만 써도 AI가 알아서 추천**")

# ===================== 전공 고정 시간표 =====================
FIXED_SCHEDULE = [
    {"name": "공학수학", "time": "화 9:00-10:15, 목 9:00-10:15", "prof": "강수진"},
    {"name": "고전읽기와토론", "time": "월 9:00-10:40", "prof": "황미은"},
    {"name": "일반화학2", "time": "월 15:00-16:15, 수 15:00-16:15", "prof": "조혜진"},
    {"name": "인공지능프로그래밍", "time": "화 13:30-14:45, 목 13:30-14:45", "prof": "이휘돈"},
    {"name": "일반물리학2", "time": "화 16:30-17:45, 목 16:30-17:45", "prof": "양하늬"},
]
AREAS = {1:"사상/역사",2:"사회/문화",3:"문학/예술",4:"과학/기술",5:"건강/레포츠",6:"외국어",7:"융복합"}

# ===================== AI 모델 =====================
@st.cache_resource
def load_model():
    with st.spinner("AI 모델 로딩 중… (최초 30초)"):
        return SentenceTransformer('jhgan/ko-sroberta-multitask')
model = load_model()

# ===================== 시간 파싱 (원본 그대로) =====================
def to_min(t): 
    try: h,m = map(int,t.split(':')); return h*60+m 
    except: return 0

def parse_time(text):
    if not isinstance(text,str): return []
    text = re.sub(r'<br/?>|\n|,', ' ', text)
    slots = []
    regex = re.compile(r"([월화수목금])\s*(\d{1,2}:\d{2})\s*(?:[-~]\s*(\d{1,2}:\d{2})|\(\s*(\d+)\s*\))")
    yoil = {"월":0,"화":1,"수":2,"목":3,"금":4}
    for m in regex.finditer(text):
        d,s,e,dur = m.group(1),m.group(2),m.group(3),m.group(4)
        start = to_min(s)
        end = to_min(e) if e else start + int(dur or 0)
        if end > start:
            slots.append({"day":yoil[d],"start":start,"end":end})
    return slots

# ===================== 데이터 로드 =====================
fixed_courses = []
for i,d in enumerate(FIXED_SCHEDULE):
    if s:=parse_time(d['time']):
        fixed_courses.append({**d,"id":f"maj_{i}","type":"major","slots":s})

courses = []
for i in range(1,8):
    f = f"section{i}.csv"
    if os.path.exists(f):
        df = pd.read_csv(f, encoding="cp949").fillna("")
        for _,r in df.iterrows():
            name = str(r.get("교과목명(미확정구분)","")).strip()
            prof = str(r.get("교수명","")).strip()
            time = str(r.get("시간/강의실",""))
            if name and (s:=parse_time(time)):
                courses.append({
                    "id":len(courses),"name":name,"prof":prof,"area":i,"type":"general",
                    "slots":s,"search_text":f"{name} {AREAS[i]} {prof}"
                })

st.success(f"전공 {len(fixed_courses)}개 + 교양 {len(courses)}개 로드 완료!")

# ===================== AI 임베딩 =====================
embeddings = model.encode([c["search_text"] for c in courses], convert_to_tensor=True)

def calc_score(kw):
    for c in courses: c["score"] = 0.0
    if not kw: return
    sims = util.cos_sim(model.encode(kw), embeddings)[0]
    for i,s in enumerate(sims):
        courses[i]["score"] = float(s)*100
    for c in courses:
        if kw in c["name"]: c["score"] += 50

def no_overlap(sched):
    slots = sorted([(s["day"],s["start"],s["end"]) for c in sched for s in c["slots"]])
    return not any(i<len(slots)-1 and slots[i][0]==slots[i+1][0] and slots[i][2]>slots[i+1][1] 
                   for i in range(len(slots)-1))

def make_timetable(areas, n, kw=""):
    calc_score(kw)
    pool = [c for c in courses if c["area"] in areas]
    if kw: pool = [c for c in pool if c["score"]>30 or kw in c["name"]]
    pool.sort(key=lambda x: -x["score"])
    pool = pool[:60]

    results = []
    for _ in range(2500):
        picks = random.sample(pool, min(len(pool), n))
        sched = fixed_courses + picks
        if no_overlap(sched):
            score = sum(c["score"] for c in picks)
            results.append({"score":score, "sched":sched})
            if len(results)>=5: break
    return sorted(results, key=lambda x:-x["score"])[:3]

# ===================== 완벽한 시간표 HTML (이제 진짜 예쁘게 나옴!) =====================
def draw_timetable(sched):
    PX = 1.3
    H_START, H_END = 9, 19
    html = f"""<style>
        .timetable {{font-family:'Malgun Gothic',sans-serif;border:1px solid #ddd;border-radius:10px;overflow:hidden;box-shadow:0 6px 20px rgba(0,0,0,0.15);margin:20px 0}}
        .days {{display:flex;background:#2c3e50;color:white;font-weight:bold}}
        .day {{flex:1;text-align:center;padding:12px}}
        .grid {{display:flex;height:{(H_END-H_START)*60*PX}px;position:relative}}
        .timecol {{width:60px;background:#f8f9fa;border-right:2px solid #2c3e50;position:relative}}
        .col {{flex:1;border-right:1px solid #eee;position:relative}}
        .slot {{position:absolute;left:4%;width:92%;padding:8px;border-radius:8px;color:white;text-align:center;
                box-shadow:4px 4px 12px rgba(0,0,0,0.3);font-weight:bold;overflow:hidden}}
        .tag {{font-size:10px;opacity:0.9;margin-bottom:4px;display:block}}
    </style>
    <div class="timetable">
        <div class="days">{"".join("<div class='day'>"+d+"</div>" for d in "월화수목금")}</div>
        <div class="grid">
            <div class="timecol">
                {"".join(f"<div style='position:absolute;top:{(h-H_START)*60*PX}px;width:100%;text-align:right;padding-right:10px;color:#555;font-size:12px'>{h:02d}:00</div>" for h in range(H_START,H_END+1))}
            </div>"""
    for day in range(5):
        html += "<div class='col'>"
        for c in sched:
            for s in c["slots"]:
                if s["day"] == day:
                    top = (s["start"] - H_START*60) * PX
                    hgt = (s["end"] - s["start"]) * PX
                    if c["type"]=="major":
                        color, tag = "#3498db", "전공"
                    elif c.get("score",0)>80:
                        color, tag = "#e74c3c", "강력추천"
                    elif c.get("score",0)>40:
                        color, tag = "#27ae60", "AI추천"
                    else:
                        color, tag = "#f39c12", AREAS.get(c["area"],"교양")
                    html += f"<div class='slot' style='top:{top}px;height:{hgt}px;background:{color}'>"
                    html += f"<div class='tag'>{tag}</div>{c['name']}<br><small style='opacity:0.9'>{c['prof']}</small></div>"
        html += "</div>"
    html += "</div></div>"
    return html

# ===================== UI =====================
c1, c2 = st.columns(2)
with c1:
    st.subheader("교양 영역")
    sel = []
    for k,v in AREAS.items():
        if st.checkbox(v, key=k): sel.append(k)
with c2:
    st.subheader("설정")
    n = st.selectbox("교양 과목 수", [1,2,3], 1)
    kw = st.text_input("AI 키워드", placeholder="예: 운동, 영어, 경제, 예술, 코딩")

if st.button("시간표 생성", type="primary"):
    if not sel:
        st.error("영역을 하나 이상 선택해주세요!")
    else:
        with st.spinner("AI가 최고의 시간표를 만들고 있어요…"):
            res = make_timetable(sel, n, kw)
        if not res:
            st.error("조건에 맞는 시간표를 못 찾았어요. 키워드를 바꾸거나 영역을 늘려보세요!")
        else:
            st.balloons()
            for i, r in enumerate(res):
                with st.expander(f"추천 {i+1}위 (AI 점수: {r['score']:.0f}점)", expanded=True):
                    for c in [x for x in r["sched"] if x["type"]=="general"]:
                        tag = "강력추천" if c.get("score",0)>80 else ("AI추천" if c.get("score",0)>40 else "")
                        st.write(f"• {c['name']} ({c['prof']}) {tag}")
                    components.html(draw_timetable(r["sched"]), height=920, scrolling=True)
