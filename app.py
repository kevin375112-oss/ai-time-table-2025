# app.py - Streamlit 최종본 (물결표 이스케이프 적용 완료)
import streamlit as st
import pandas as pd
import os
import re
import random
import numpy as np
from sentence_transformers import SentenceTransformer, util
import streamlit.components.v1 as components
import time

# ===================== [CSS 로드] =====================
# 외부 CSS를 로드하여 취소선(text-decoration) 문제를 강제로 해결합니다.
try:
    timestamp = time.time()
    with open("styles.css") as f:
        st.markdown(f'<style href="styles.css?t={timestamp}">{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.markdown("""
        <style>
        * { text-decoration: none !important; }
        span, div, p, a, strong, b, em { text-decoration: none !important; }
        </style>
    """, unsafe_allow_html=True)
    
# ===================== [설정] =====================
FIXED_SCHEDULE = [
    {"name": "공학수학", "time": "화 9:00-10:15 507-101, 목 9:00-10:15 507-101", "prof": "강수진"},
    {"name": "고전읽기와토론", "time": "월 9:00-10:40 311-104", "prof": "황미은"},
    {"name": "일반화학2", "time": "월 15:00-16:15 507-101, 수 15:00-16:15 507-101", "prof": "조혜진"},
    {"name": "인공지능프로그래밍", "time": "화 13:30-14:45 314-204-2, 목 13:30-14:45 314-204-2", "prof": "이휘돈"},
    {"name": "일반물리학2", "time": "화 16:30-17:45 507-102, 목 16:30-17:45 507-102", "prof": "양하늬"},
]
AREAS = {1:"사상/역사", 2:"사회/문화", 3:"문학/예술", 4:"과학/기술", 5:"건강/레포츠", 6:"외국어", 7:"융복합"}
FILE_LIST = [("section1.csv",1),("section2.csv",2),("section3.csv",3),("section4.csv",4),
             ("section5.csv",5),("section6.csv",6),("section7.csv",7)]
COLS = {'name':'교과목명(미확정구분)', 'time':'시간/강의실', 'prof':'교수명', 'rate':'교양평점'}

# ===================== [로직 1] 데이터 파싱 및 로드 =====================
@st.cache_resource
def load_model():
    with st.spinner("🤖 AI 모델 로딩 중..."):
        return SentenceTransformer('jhgan/ko-sroberta-multitask')
model = load_model()

def parse_data(raw_str):
    """ 시간/장소/슬롯 데이터 추출 (물결표 문제 해결) """
    if not isinstance(raw_str, str): return [], "", ""
    
    parts = [p.strip() for p in re.sub(r'<br/?>|\n', ',', raw_str).split(',') if p.strip()]
    slots, fmt_times, rooms = [], [], []
    last_day = None
    yoil_map = {d:i for i,d in enumerate("월화수목금토일")}
    
    p_rng = re.compile(r"([월화수목금토일])?\s*(\d{1,2}:\d{2})\s*[-~]\s*(\d{1,2}:\d{2})(.*)")
    p_dur = re.compile(r"([월화수목금토일])?\s*(\d{1,2}:\d{2})\s*\(\s*(\d+)\s*\)(.*)")

    def to_min(t):
        try: h, m = map(int, t.split(':')); return h*60 + m
        except: return 0

    for p in parts:
        d_str, start, dur, extra = None, 0, 0, ""
        s_str_used = ""  
        end = 0
        
        if m := p_rng.search(p): 
            d_str, s_str_raw, e_str_raw, extra = m.groups() 
            start = to_min(s_str_raw)
            end = to_min(e_str_raw)
            dur = end - start
            s_str_used = s_str_raw
        elif m := p_dur.search(p): 
            d_str, s_str_raw, dur_str, extra = m.groups() 
            start = to_min(s_str_raw)
            dur = int(dur_str)
            end = start + dur
            s_str_used = s_str_raw
        else:
            rooms.append(p)
            continue

        if d_str: last_day = d_str
        if not last_day or dur <= 0: continue
        
        end_time_str = f"{end // 60:02d}:{end % 60:02d}"
        
        slots.append({'day': yoil_map[last_day], 'start': start, 'end': end})
        fmt_times.append(f"{last_day} {s_str_used}~{end_time_str}") 
        
        if extra and extra.strip(): rooms.append(extra.strip())

    room_str = ", ".join(sorted(list(set(rooms))))
    if not room_str: room_str = ""
    
    return slots, ", ".join(fmt_times), room_str

# 데이터 로드
fixed_courses = []
for i, d in enumerate(FIXED_SCHEDULE):
    s, t, r = parse_data(d['time'])
    if s: fixed_courses.append({**d, 'id':f"maj_{i}", 'area':'전공', 'rating':0.0, 'slots':s, 'type':'major', 'time_str':t, 'room':r})

courses = []
for fname, area in FILE_LIST:
    if not os.path.exists(fname): continue
    try:
        enc = 'cp949' if fname.endswith('.csv') else None
        try: df = pd.read_csv(fname, encoding=enc).fillna('') if enc else pd.read_excel(fname).fillna('')
        except: df = pd.read_csv(fname, encoding='euc-kr').fillna('')
        for _, r in df.iterrows():
            try: rating = float(r.get(COLS['rate']))
            except: rating = 0.0
            s, t, r_str = parse_data(str(r.get(COLS['time'])))
            if s:
                c_name = str(r.get(COLS['name'])).strip()
                courses.append({
                    'id': len(courses), 'name': c_name, 
                    'prof': str(r.get(COLS['prof'])).strip(),
                    'rating': rating, 'area': area, 'slots': s, 'type': 'general', 
                    'time_str': t, 'room': r_str, 'search_text': c_name, 
                    'match_score': 0.0
                })
    except Exception as e: st.error(f"Error loading {fname}: {e}")

st.sidebar.success(f"✅ 전공 {len(fixed_courses)}개, 교양 {len(courses)}개 로드 완료")

# AI 벡터화
@st.cache_data
def get_course_embeddings(courses_list):
    st.sidebar.info("🔄 데이터 분석 중...")
    embeddings = model.encode([c['search_text'] for c in courses_list], convert_to_tensor=True)
    st.sidebar.success("✅ 분석 완료")
    return embeddings
course_embeddings = get_course_embeddings(courses)

# ===================== [로직 2] AI 엔진 & 스케줄링 =====================
def calc_score(keyword, courses_list):
    for c in courses_list: c['match_score'] = 0.0
    if not keyword: return
    q_vec = model.encode(keyword, convert_to_tensor=True)
    sims = util.cos_sim(q_vec, course_embeddings)[0].cpu().numpy()
    for i, s in enumerate(sims):
        c = courses_list[i]
        c['match_score'] = float(s) * 100
        if keyword in c['name']: c['match_score'] += 30 

def check_collision(sched):
    slots = sorted([(s['day'], s['start'], s['end']) for c in sched for s in c['slots']])
    return any(slots[i][0] == slots[i+1][0] and slots[i][2] > slots[i+1][1] for i in range(len(slots)-1))

def run_ai(target_areas, pick_n, keyword=""):
    temp_courses = [c.copy() for c in courses]
    calc_score(keyword, temp_courses)
    pool = [c for c in temp_courses if c['area'] in target_areas and not any(s['day']==4 for s in c['slots'])]
    if keyword:
        filtered = [c for c in pool if c['match_score'] > 40]
        if not filtered: return []
        pool = filtered
    pool.sort(key=lambda x: -(x['match_score']*5 + x['rating']))
    pool = pool[:60]
    results = []
    for _ in range(1000): 
        curr = fixed_courses[:]
        picks = random.sample(pool, min(len(pool), pick_n))
        valid = True
        for p in picks:
            if any(p['name'] == c['name'] for c in curr) or check_collision(curr + [p]):
                valid = False; break
            curr.append(p)
        if valid and len(curr) == len(fixed_courses) + pick_n:
            score = sum(c['match_score']*5 + c['rating'] for c in picks)
            ids = tuple(sorted(c['id'] for c in curr if c['type']=='general'))
            results.append({'score': score, 'schedule': curr, 'ids': ids})
    unique = {r['ids']: r for r in results}.values()
    return sorted(unique, key=lambda x: -x['score'])[:3]

# ===================== [로직 3] 시각화 & UI =====================
def render_timetable(sched):
    PX = 1.3; H_S = 9; H_E = 22  
    TOTAL_H = (H_E - H_S) * 60 * PX
    
    html = f"""
    <style>
        .tt-con {{ display:flex; font-family:'Malgun Gothic'; font-size:12px; border:1px solid #ccc; width:100%; }}
        .tt-col {{ position:relative; border-right:1px solid #eee; height:{TOTAL_H}px; flex:1; }}
        .tt-tm {{ width:60px; background:#fafafa; border-right:1px solid #ccc; position:relative; height:{TOTAL_H}px; }}
        /* time label의 border-top 제거 */
        .tt-lbl {{ position:absolute; width:100%; text-align:right; padding-right:5px; font-size:11px; color:#888; border-top:none; }} 
        .tt-grd {{ position:absolute; width:100%; border-top:1px solid #f4f4f4; }}
        /* 강의 카드에 z-index를 부여하여 격자선 위에 표시 */
        .tt-crd {{ position:absolute; width:94%; left:3%; padding:2px; border-radius:4px; box-sizing:border-box; 
                   font-size:10px; line-height:1.2; box-shadow:1px 1px 3px #ddd; display:flex; flex-direction:column; justify-content:center; text-align:center; 
                   z-index: 10; }} 
    </style>
    <div style='display:flex; margin-left:60px;'>
        {''.join([f"<div style='flex:1; text-align:center; padding:5px; background:#f0f0f0; font-weight:bold; border-right:1px solid #fff;'>{d}</div>" for d in "월화수목금"])}
    </div>
    <div class='tt-con'>
        <div class='tt-tm'>
            {''.join([f"<div class='tt-lbl' style='top:{(h-H_S)*60*PX}px; height:{60*PX}px'>{h:02d}:00</div>" for h in range(H_S, H_E)])}
        </div>
    """
    for d in range(5):
        html += "<div class='tt-col'>"
        
        # 정시 가로선 (격자선)
        html += ''.join([f"<div class='tt-grd' style='top:{(h-H_S)*60*PX}px;'></div>" for h in range(H_S, H_E)])
        
        for c in sched:
            for s in c['slots']:
                if s['day'] == d:
                    top = (s['start'] - H_S*60) * PX
                    hgt = (s['end'] - s['start']) * PX
                    
                    if c['type']=='major':
                        sty = ("#e3f2fd","#2196f3","#0d47a1","전공")
                    else:
                        sty = ("#fff3e0","#ff9800","#e65100", AREAS.get(c['area'],'교양'))
                        if c.get('match_score',0)>60: sty = ("#e8f5e9","#4caf50","#1b5e20","AI추천")
                        
                    info = f"<span style='font-size:9px; color:{sty[2]};'>({c.get('room','N/A')})</span>"
                    
                    # text-decoration: none;을 포함하여 취소선 방지
                    time_info = f"<span style='font-size:9px; color:{sty[2]}; text-decoration: none;'>{s['start']//60:02d}:{s['start']%60:02d}~{s['end']//60:02d}:{s['end']%60:02d}</span>"
                    
                    html += f"""<div class='tt-crd' style='top:{top}px; height:{hgt}px; background:{sty[0]}; border-left:4px solid {sty[1]}; color:{sty[2]};'>
                                 <span style='font-size:9px; background:rgba(255,255,255,0.7); padding:1px 4px; border-radius:3px;'>{sty[3]}</span>
                                 <b>{c['name']}</b><br><span style='font-size:9px;'>{c['prof']}</span><br>{time_info}<br>{info}</div>"""
        html += "</div>"
    html += "</div>"
    return html

# ===================== Streamlit UI =====================
st.set_page_config(page_title="AI 스마트 시간표", layout="wide")
st.title("🧠 AI 스마트 시간표 생성기")
st.markdown("**전공 고정 │ 시간 겹침 0% │ 깔끔한 그리드**")

col_settings, col_areas = st.columns([1, 1.5])

with col_areas:
    st.subheader("📚 영역 선택")
    selected_areas = []
    cols = st.columns(2)
    for i, (k, v) in enumerate(AREAS.items()):
        if cols[i % 2].checkbox(v, key=f"area_{k}", value=False):
            selected_areas.append(k)

with col_settings:
    st.subheader("⚙️ 설정")
    num_courses = st.selectbox("교양 과목 수", [1, 2, 3], index=1, key='num')
    keyword = st.text_input("AI 검색 키워드 (선택)", placeholder="예: 경제, 운동, 영어", key='key')
    st.markdown("---")
    generate_button = st.button("시간표 생성 🚀", type="primary", use_container_width=True)

st.markdown("---")

if generate_button:
    if not selected_areas:
        st.error("⚠️ 영역을 하나 이상 선택해주세요!")
    else:
        st.info("💡 **주의:** 브라우저에 문제가 있는 경우, **Ctrl + Shift + R**을 눌러 강제 새로고침을 시도해주세요.")
        
        with st.spinner("⏳ AI가 최적의 시간표를 분석하고 있습니다..."):
            res = run_ai(selected_areas, num_courses, keyword)
        
        if not res:
            st.error("❌ 선택한 조건에 맞는 시간표 조합을 찾을 수 없습니다. (금요일 제외 조건 때문일 수 있습니다.)")
        else:
            st.success(f"✅ 총 {len(res)}개의 추천 시간표를 찾았습니다.")
            
            for i, r in enumerate(res):
                match = any(c.get('match_score', 0) > 60 for c in r['schedule'] if c['type'] == 'general')
                title = f"추천 {i+1}위 " + ("(🎯 AI 적중)" if match else "(평점 우수)")
                
                with st.expander(title, expanded=(i == 0)):
                    st.markdown("### 선택된 교양 과목 목록")
                    for c in r['schedule']:
                        if c['type'] == 'general':
                            tag = "✨AI" if c.get('match_score', 0) > 60 else ""
                            
                            # 💡 물결표 이스케이프 처리: 출력 시 물결표가 그대로 보이도록 \~로 치환
                            time_str_safe = c['time_str'].replace('~', '\~')

                            st.markdown(
                                f"""
                                • **{c['name']}** ({c['prof']}) | 평점: **{c['rating']:.2f}** {tag} | 시간: **{time_str_safe}** | 강의실: {c.get('room','N/A')}
                                """
                            )
                    
                    st.markdown("### 시간표 시각화")
                    components.html(render_timetable(r['schedule']), height=850, scrolling=True)
