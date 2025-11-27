# app.py — Jupyter와 100% 동일하게 작동하는 Streamlit 최종본
import streamlit as st
import pandas as pd
import os
import re
import random
from sentence_transformers import SentenceTransformer, util

# ===================== 1. Streamlit Cloud에서 CSV 강제 복사 =====================
for i in range(1, 8):
    src = f"/mount/src/ai-time-table-2025/section{i}.csv"
    dst = f"section{i}.csv"
    if os.path.exists(src) and not os.path.exists(dst):
        os.system(f"cp {src} {dst}")

st.set_page_config(page_title="2025 AI 시간표", layout="wide")
st.title("2025-2학기 AI 시간표 생성기")
st.caption("Jupyter Notebook과 정확히 동일한 로직으로 작동합니다")

# ===================== 2. 전공 고정 시간표 =====================
FIXED_SCHEDULE = [
    {"name": "공학수학", "time": "화 9:00-10:15, 목 9:00-10:15", "prof": "강수진"},
    {"name": "고전읽기와토론", "time": "월 9:00-10:40", "prof": "황미은"},
    {"name": "일반화학2", "time": "월 15:00-16:15, 수 15:00-16:15", "prof": "조혜진"},
    {"name": "인공지능프로그래밍", "time": "화 13:30-14:45, 목 13:30-14:45", "prof": "이휘돈"},
    {"name": "일반물리학2", "time": "화 16:30-17:45, 목 16:30-17:45", "prof": "양하늬"},
]
AREAS = {1:"사상/역사",2:"사회/문화",3:"문학/예술",4:"과학/기술",5:"건강/레포츠",6:"외국어",7:"융복합"}

# ===================== 3. AI 모델 로딩 =====================
@st.cache_resource
def load_model():
    with st.spinner("AI 모델 로딩 중… (최초 30초 정도)"):
        return SentenceTransformer('jhgan/ko-sroberta-multitask')
model = load_model()
st.success("AI 모델 준비 완료!")

# ===================== 4. 원본 그대로 시간 파싱 =====================
def to_min(t_str):
    try: h, m = map(int, t_str.split(':')); return h*60 + m
    except: return 0

def parse_time(text):
    if not isinstance(text, str): return []
    text = re.sub(r'<br/?>|\n|,', ' ', text)
    slots = []
    regex = re.compile(r"([월화수목금토일])\s*(\d{1,2}:\d{2})\s*(?:[-~]\s*(\d{1,2}:\d{2})|\(\s*(\d+)\s*\))")
    yoil_map = {d:i for i,d in enumerate("월화수목금토일")}
    for m in regex.finditer(text):
        d_str, s_str, e_str, dur_str = m.groups()
        start = to_min(s_str)
        end = to_min(e_str) if e_str else start + int(dur_str or 0)
        if end > start:
            slots.append({'day': yoil_map.get(d_str, 0), 'start': start, 'end': end})
    return slots

# ===================== 5. 데이터 로드 (원본 그대로) =====================
fixed_courses = []
for i, d in enumerate(FIXED_SCHEDULE):
    if s := parse_time(d['time']):
        fixed_courses.append({**d, 'id':f"maj_{i}", 'area':'전공', 'rating':5.0, 'slots':s, 'type':'major', 'time_str':d['time']})

courses = []
for fname, area in [("section1.csv",1),("section2.csv",2),("section3.csv",3),("section4.csv",4),("section5.csv",5),("section6.csv",6),("section7.csv",7)]:
    if os.path.exists(fname):
        df = pd.read_csv(fname, encoding='cp949').fillna('')
        for _, r in df.iterrows():
            try: rating = float(r.get('교양평점', 0))
            except: rating = 0.0
            raw_time = str(r.get('시간/강의실', ''))
            if s := parse_time(raw_time):
                c_name = str(r.get('교과목명(미확정구분)', '')).strip()
                c_prof = str(r.get('교수명', '')).strip()
                courses.append({
                    'id': len(courses), 'name': c_name, 'prof': c_prof, 'rating': rating,
                    'area': area, 'slots': s, 'type': 'general', 'time_str': raw_time.split(',')[0],
                    'match_score': 0.0,
                    'search_text': f"{c_name} {AREAS.get(area,'')}"
                })

st.write(f"전공 {len(fixed_courses)}개, 교양 {len(courses)}개 로드 완료")

# ===================== 6. AI 벡터 미리 계산 (원본 그대로) =====================
course_embeddings = model.encode([c['search_text'] for c in courses], convert_to_tensor=True)

# ===================== 7. 원본 그대로 AI 엔진 =====================
def calculate_scores(user_keyword):
    for c in courses: c['match_score'] = 0.0
    if not user_keyword: return
    query_embedding = model.encode(user_keyword, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, course_embeddings)[0]
    for i, score in enumerate(cos_scores):
        courses[i]['match_score'] = float(score) * 100.0
    for c in courses:
        if user_keyword in c['name']:
            c['match_score'] += 50.0

def check_overlap(sched):
    slots = sorted([(s['day'], s['start'], s['end']) for c in sched for s in c['slots']])
    return any(slots[i][0] == slots[i+1][0] and slots[i][2] > slots[i+1][1] for i in range(len(slots)-1))

def run_ai(target_areas, pick_n, user_keyword=""):
    calculate_scores(user_keyword)
    pool = [c for c in courses if c['area'] in target_areas]
    if user_keyword:
        pool = [c for c in pool if c['match_score'] > 30.0 or user_keyword in c['name']]
    pool.sort(key=lambda x: -(x['match_score'] * 5 + x['rating']))
    pool = pool[:50]

    results = []
    for _ in range(2000):
        picks = random.sample(pool, min(len(pool), pick_n))
        curr = fixed_courses[:]
        valid = True
        for p in picks:
            if any(p['name'] == c['name'] for c in curr) or check_overlap(curr + [p]):
                valid = False; break
            curr.append(p)
        if valid:
            total_score = sum(c['match_score'] for c in picks) * 5 + sum(c['rating'] for c in picks)
            ids = tuple(sorted(c['id'] for c in picks))
            results.append({'score': total_score, 'schedule': curr, 'ids': ids})
    unique = {r['ids']: r for r in results}.values()
    return sorted(unique, key=lambda x: -x['score'])[:3]

# ===================== 8. 원본 그대로 HTML 시각화 =====================
def render_html(sched):
    PX = 1.3
    H_START, H_END = 9, 19
    TOTAL_H = (H_END - H_START) * 60 * PX
    html = f"""<style>
        .tt-box {{display:flex;font-family:'Malgun Gothic',sans-serif;font-size:12px;border:1px solid #ddd;width:100%}}
        .tt-col {{position:relative;border-right:1px solid #eee;height:{TOTAL_H}px;flex:1}}
        .tt-card {{position:absolute;width:94%;left:3%;padding:4px;border-radius:4px;box-sizing:border-box;
                   font-size:11px;line-height:1.3;box-shadow:1px 1px 3px rgba(0,0,0,0.1);word-break:break-all;white-space:normal}}
        .tt-badge {{font-size:9px;padding:1px 3px;border-radius:3px;margin-bottom:2px}}
    </style>
    <div style='display:flex;margin-left:60px'>
        {''.join(f"<div style='flex:1;text-align:center;padding:5px;background:#333;color:white;font-weight:bold;border-right:1px solid #fff'>{d}</div>" for d in "월화수목금")}
    </div>
    <div class='tt-box'>
        <div style='width:60px;background:#fafafa;border-right:1px solid #ddd;position:relative;height:{TOTAL_H}px'>
            {''.join(f"<div style='position:absolute;top:{(h-H_START)*60*PX}px;width:100%;text-align:right;padding-right:5px;font-size:11px;color:#888;border-top:1px solid #eee'>{h:02d}:00</div>" for h in range(H_START, H_END))}
        </div>"""
    for d_idx in range(5):
        html += "<div class='tt-col'>"
        for c in sched:
            for s in c['slots']:
                if s['day'] == d_idx:
                    top = (s['start'] - H_START*60) * PX
                    hgt = (s['end'] - s['start']) * PX
                    if c['type']=='major':
                        bg, bd, txt, tag = "#e3f2fd","#2196f3","#0d47a1","전공"
                    elif c.get('match_score',0)>80:
                        bg, bd, txt, tag = "#ffebee","#f44336","#b71c1c","강력추천"
                    elif c.get('match_score',0)>40:
                        bg, bd, txt, tag = "#e8f5e9","#4caf50","#1b5e20","AI추천"
                    else:
                        bg, bd, txt, tag = "#fff3e0","#ff9800","#e65100",AREAS.get(c['area'],'교양')
                    html += f"<div class='tt-card' style='top:{top}px;height:{hgt}px;background:{bg};border-left:4px solid {bd};color:{txt}'>"
                    html += f"<div class='tt-badge' style='background:rgba(0,0,0,0.1)'>{tag}</div>"
                    html += f"<b>{c['name']}</b><br>{c['prof']}</div>"
        html += "</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ===================== 9. UI (Jupyter와 동일한 동작) =====================
col1, col2 = st.columns(2)
with col1:
    st.subheader("영역 선택")
    area_checks = {}
    for k, v in AREAS.items():
        area_checks[k] = st.checkbox(v, key=f"area_{k}")

with col2:
    st.subheader("설정")
    num = st.selectbox("교양 과목 수", [1, 2, 3], 1)
    keyword = st.text_input("AI 검색 (예: 운동, 영어, 경제)", "")

if st.button("시간표 생성", type="primary"):
    sel = [k for k, v in area_checks.items() if v]
    if not sel:
        st.error("영역을 하나 이상 선택해주세요!")
    else:
        with st.spinner("AI가 시간표 만드는 중…"):
            res = run_ai(sel, num, keyword)
        if not res:
            st.error("조건에 맞는 시간표를 찾지 못했습니다.")
        else:
            for i, r in enumerate(res):
                high = any(c.get('match_score',0)>40 for c in r['schedule'] if c['type']=='general')
                with st.expander(f"추천 {i+1}위 {'(AI 매칭 성공!)' if high else ''}"):
                    for c in r['schedule']:
                        if c['type']=='general':
                            score = c.get('match_score',0)
                            tag = f"강력추천({int(score)}%)" if score>80 else (f"AI매칭({int(score)}%)" if score>40 else "")
                            st.write(f"• {c['name']} ({c['prof']}) {tag}")
                    render_html(r['schedule'])
