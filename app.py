import streamlit as st
import pandas as pd
import re, random
from sentence_transformers import SentenceTransformer, util
import os

st.title("2025-2학기 AI 시간표 생성기")
st.write("CSV 파일 7개 올라가면 바로 사용 가능!")

# 모델 로딩
@st.cache_resource
def get_model():
    return SentenceTransformer('jhgan/ko-sroberta-multitask')
model = get_model()

# 시간 파싱
def parse_time(t):
    if not isinstance(t, str): return []
    slots = []
    for m in re.finditer(r"([월화수목금])\s*(\d{1,2}:\d{2})[-~]\s*(\d{1,2}:\d{2})", t):
        day = "월화수목금".index(m.group(1))
        s = int(m.group(2).split(":")[0])*60 + int(m.group(2).split(":")[1])
        e = int(m.group(3).split(":")[0])*60 + int(m.group(3).split(":")[1])
        slots.append({'day':day, 'start':s, 'end':e})
    return slots

# 데이터 로드
courses = []
for i in range(1,8):
    f = f"section{i}.csv"
    if os.path.exists(f):
        df = pd.read_csv(f, encoding='cp949').fillna('')
        for _, r in df.iterrows():
            name = str(r.get('교과목명(미확정구분)', ''))
            prof = str(r.get('교수명', ''))
            time = str(r.get('시간/강의실', ''))
            if s := parse_time(time):
                courses.append({
                    'name': name, 'prof': prof, 'area': i,
                    'slots': s, 'search': name + " " + ["","사상역사","사회문화","문학예술","과학기술","건강레포츠","외국어","융복합"][i]
                })

if not courses:
    st.error("CSV 파일을 찾을 수 없어요! section1.csv ~ section7.csv 확인해주세요")
    st.stop()

embeddings = model.encode([c['search'] for c in courses], convert_to_tensor=True)

st.success(f"총 {len(courses)}개 교양 과목 로드 완료!")

sel = st.multiselect("영역 선택", ["사상/역사","사회/문화","문학/예술","과학/기술","건강/레포츠","외국어","융복합"])
num = st.slider("원하는 교양 수", 1, 3, 2)
kw = st.text_input("AI 키워드 (예: 운동, 영어, 경제)", "")

if st.button("시간표 만들어줘!", type="primary"):
    if not sel: st.warning("영역을 선택해주세요")
    else:
        score = util.cos_sim(model.encode(kw or " "), embeddings)[0] if kw else [1]*len(courses)
        for i,c in enumerate(courses):
            c['score'] = float(score[i]) * 100
        
        pool = [c for c in courses if ["","사상역사","사회문화","문학예술","과학기술","건강레포츠","외국어","융복합"][c['area']] in sel]
        pool.sort(key=lambda x: -x['score'])
        
        for _ in range(1000):
            picks = random.sample(pool[:50], min(len(pool), num))
            all_ok = True
            for a in picks:
                for b in picks:
                    if a is b: continue
                    if a['slots'][0]['day'] == b['slots'][0]['day'] and max(a['slots'][0]['start'], b['slots'][0]['start']) < min(a['slots'][0]['end'], b['slots'][0]['end25']):
                        all_ok = False
            if all_ok:
                st.success("성공!")
                for c in picks:
                    st.write(f"**{c['name']}** ({c['prof']})")
                st.balloons()
                break
        else:
            st.error("조건에 맞는 조합을 못 찾았어요")
