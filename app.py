# app.py — 15분 빈 시간 정확히 표시되는 진짜 최종본 (평점 표시 추가)
import streamlit as st
import pandas as pd
import os
import re
import random
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components

# ===================== CSV 강제 로드 =====================
for i in range(1, 8):
    src = f"/mount/src/ai-time-table-2025/section{i}.csv"
    dst = f"section{i}.csv"
    if os.path.exists(src) and not os.path.exists(dst):
        os.system(f"cp {src} {dst}")

st.set_page_config(page_title="2025 AI 시간표 생성기", layout="wide")
st.title("🧠 2025-2학기 AI 시간표 생성기")
st.markdown("**전공 고정 │ 시간 겹침 0% │ 15분 빈 시간도 정확히 표시 │ 키워드만 쳐도 AI 추천**")

# ===================== 전공 고정 =====================
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

# ===================== 시간 파싱 =====================
def to_min(t): 
    try: h,m = map(int,t.split(':')); return h*60+m 
    except: return 0

def parse_time(text):
    if not isinstance(text,str): return []
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

# ===================== 데이터 로드 =====================
fixed_courses = []
for i, d in enumerate(FIXED_SCHEDULE):
    if s := parse_time(d['time']):
        fixed_courses.append({**d, 'id':f"maj_{i}", 'area':'전공', 'rating':5.0, 'slots':s, 'type':'major'})

courses = []
for fname, area in [("section1.csv",1),("section2.csv",2),("section3.csv",3),("section4.csv",4),
                    ("section5.csv",5),("section6.csv",6),("section7.csv",7)]:
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
                    'area': area, 'slots': s, 'type': 'general',
                    'search_text': f"{c_name} {AREAS.get(area,'')}"
                })

st.success(f"전공 {len(fixed_courses)}개 + 교양 {len(courses)}개 로드 완료!")

# ===================== AI 벡터 =====================
course_embeddings = model.encode([c['search_text'] for c in courses], convert_to_tensor=True)

# ===================== AI 엔진 =====================
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
        pool = [c for c in pool if c['match_score'] > 30.0]
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
            # AI 매칭 점수 * 5 + 평점 * 1 (기존 가중치 유지)
            total_score = sum(c['match_score'] for c in picks) * 5 + sum(c['rating'] for c in picks)
            ids = tuple(sorted(c['id'] for c in picks))
            results.append({'score': total_score, 'schedule': curr, 'ids': ids})
    unique = {r['ids']: r for r in results}.values()
    return sorted(unique, key=lambda x: -x['score'])[:3]

# ===================== 완벽한 시간표 (15분 빈 시간 정확히 표시 + 가로선) =====================
def render_timetable(schedule):
    PX = 1.3
    H_START, H_END = 9, 19
    TOTAL_H = (H_END - H_START) * 60 * PX

    # 하루에 모든 슬롯 모으기 (빈 시간 계산용)
    day_slots = [[] for _ in range(5)]
    for c in schedule:
        for s in c['slots']:
            day_slots[s['day']].append((s['start'], s['end'], c))

    html = f"""<style>
        .tt-box {{display:flex;font-family:'Malgun Gothic',sans-serif;font-size:12px;border:1px solid #ddd;width:100%}}
        .tt-col {{position:relative;border-right:1px solid #eee;height:{TOTAL_H}px;flex:1;background:#fafafa}}
        .tt-card {{position:absolute;width:94%;left:3%;padding:5px;border-radius:6px;box-sizing:border-box;
                   font-size:11px;line-height:1.3;box-shadow:2px 2px 8px rgba(0,0,0,0.2);text-align:center}}
        .tt-badge {{font-size:9px;padding:2px 6px;border-radius:4px;margin-bottom:3px;background:rgba(255,255,255,0.8)}}
        .hour-line {{position:absolute;left:0;width:100%;height:1px;background:#ccc;z-index:1}}
    </style>
    <div style='display:flex;margin-left:60px;margin-bottom:10px'>
        {"".join(f"<div style='flex:1;text-align:center;padding:8px;background:#2c3e50;color:white;font-weight:bold'>{d}</div>" for d in "월화수목금")}
    </div>
    <div class='tt-box'>
        <div style='width:60px;background:#f8f9fa;border-right:2px solid #2c3e50;position:relative'>
            {"".join(f"<div style='position:absolute;top:{(h-H_START)*60*PX}px;width:100%;text-align:right;padding-right:10px;font-size:11px;color:#555'>{h:02d}:00</div>" for h in range(H_START, H_END+1))}
        </div>"""

    for day in range(5):
        html += "<div class='tt-col'>"
        # 가로선
        for h in range(H_START, H_END):
            html += f"<div class='hour-line' style='top:{(h-H_START)*60*PX}px'></div>"
        # 강의 카드 (빈 시간은 자동으로 비워짐)
        for c in schedule:
            for s in c['slots']:
                if s['day'] == day:
                    top = (s['start'] - H_START*60) * PX
                    hgt = (s['end'] - s['start']) * PX
                    if c['type']=='major':
                        bg, bd, txt, tag = "#e3f2fd","#1976d2","#0d47a1","전공"
                    elif c.get('match_score',0)>80:
                        bg, bd, txt, tag = "#ffebee","#d32f2f","#b71c1c","강력추천"
                    elif c.get('match_score',0)>40:
                        bg, bd, txt, tag = "#e8f5e9","#388e3c","#1b5e20","AI추천"
                    else:
                        bg, bd, txt, tag = "#fff3e0","#f57c00","#e65100",AREAS.get(c['area'],'교양')
                    html += f"<div class='tt-card' style='top:{top}px;height:{hgt}px;background:{bg};border-left:5px solid {bd};color:{txt}'>"
                    html += f"<div class='tt-badge'>{tag}</div><b>{c['name']}</b><br><small>{c['prof']}</small></div>"
        html += "</div>"
    html += "</div>"
    return html

# ===================== UI =====================
col1, col2 = st.columns(2)
with col1:
    st.subheader("영역 선택")
    selected = []
    for k, v in AREAS.items():
        if st.checkbox(v, key=f"a{k}"):
            selected.append(k)
with col2:
    st.subheader("설정")
    num = st.selectbox("교양 과목 수", [1,2,3], index=1)
    keyword = st.text_input("AI 검색", placeholder="예: 운동, 영어, 경제, 코딩")

if st.button("시간표 생성", type="primary"):
    if not selected:
        st.error("영역을 하나 이상 선택해주세요!")
    else:
        with st.spinner("AI가 시간표 만드는 중…"):
            res = run_ai(selected, num, keyword)
        if not res:
            st.error("조건에 맞는 시간표가 없습니다.")
        else:
            for i, r in enumerate(res):
                high = any(c.get('match_score',0)>40 for c in r['schedule'] if c['type']=='general')
                with st.expander(f"추천 {i+1}위 {'(AI 매칭 성공!)' if high else ''}", expanded=True):
                    for c in r['schedule']:
                        if c['type']=='general':
                            score = c.get('match_score', 0)
                            rating = c.get('rating', 0.0) # 평점 가져오기
                            tag = f"강력추천({int(score)}%)" if score > 80 else (f"AI매칭({int(score)}%)" if score > 40 else "")
                            
                            # 평점 정보와 AI 매칭 태그를 함께 표시
                            st.write(f"• **{c['name']}** ({c['prof']}) | 평점: **{rating:.2f}** {tag}")

                    components.html(render_timetable(r['schedule']), height=950, scrolling=True)
