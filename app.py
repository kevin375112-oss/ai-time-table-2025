import streamlit as st
import pandas as pd
import os
import re
import random
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────
# 1. Streamlit Cloud에서 CSV 강제 로드
# ─────────────────────────────────────
for i in range(1, 8):
    src = f"/mount/src/ai-time-table-2025/section{i}.csv"
    dst = f"section{i}.csv"
    if os.path.exists(src) and not os.path.exists(dst):
        os.system(f"cp {src} {dst}")

# ─────────────────────────────────────
# 2. 기본 설정
# ─────────────────────────────────────
st.set_page_config(page_title="2025 AI 시간표 생성기", layout="wide")
st.title("2025-2학기 AI 시간표 생성기")
st.markdown("**전공 고정 + 교양 자동 추천 + 시간 절대 안 겹침 + AI 키워드 검색**")

# ─────────────────────────────────────
# 3. AI 모델 로딩
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    st.write("AI 모델 로딩 중… (최초 1회만 걸려요)")
    return SentenceTransformer('jhgan/ko-sroberta-multitask')
model = load_model()

# ─────────────────────────────────────
# 4. 시간 파싱 함수
# ─────────────────────────────────────
def parse_time(text):
    if not isinstance(text, str): return []
    slots = []
    pattern = r"([월화수목금])\s*(\d{1,2}:\d{2})[-~]\s*(\d{1,2}:\d{2})"
    for m in re.finditer(pattern, text):
        day = "월화수목금".index(m.group(1))
        s = int(m.group(2).split(":")[0])*60 + int(m.group(2).split(":")[1])
        e = int(m.group(3).split(":")[0])*60 + int(m.group(3).split(":")[1])
        slots.append({"day": day, "start": s, "end": e})
    return slots

# ─────────────────────────────────────
# 5. CSV 데이터 로드
# ─────────────────────────────────────
courses = []
for i in range(1, 8):
    file = f"section{i}.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, encoding="cp949").fillna("")
        for _, row in df.iterrows():
            name = str(row.get("교과목명(미확정구분)", "")).strip()
            prof = str(row.get("교수명", "")).strip()
            time = str(row.get("시간/강의실", ""))
            s = parse_time(time)  # := 대신 이렇게 수정
            if name and s:  # 조건 분리
                courses.append({
                    "name": name,
                    "prof": prof,
                    "area": i,
                    "slots": s,
                    "search": name + " " + ["","사상/역사","사회/문화","문학/예술","과학/기술","건강/레포츠","외국어","융복합"][i]
                })

if not courses:
    st.error("CSV 파일을 찾을 수 없어요! section1.csv ~ section7.csv 확인")
    st.stop()

st.success(f"총 {len(courses)}개 교양 과목 로드 완료!")

# 임베딩 미리 계산
embeddings = model.encode([c["search"] for c in courses], convert_to_tensor=True)

# ─────────────────────────────────────
# 6. UI
# ─────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("교양 영역 선택")
    areas = st.multiselect("복수 선택 가능", 
        ["사상/역사","사회/문화","문학/예술","과학/기술","건강/레포츠","외국어","융복합"],
        default=["건강/레포츠", "외국어"])
with col2:
    st.subheader("설정")
    num = st.slider("원하는 교양 과목 수", 1, 3, 2)
    keyword = st.text_input("AI 키워드 검색 (예: 운동, 영어, 경제, 철학)", "")

if st.button("시간표 만들어줘!", type="primary"):
    if not areas:
        st.warning("영역을 하나 이상 선택해주세요!")
    else:
        with st.spinner("AI가 최적 시간표 찾는 중…"):
            # 키워드 점수 계산
            if keyword:
                sims = util.cos_sim(model.encode(keyword), embeddings)[0]
                for i, c in enumerate(courses):
                    c["score"] = float(sims[i])
            else:
                for c in courses: c["score"] = 1.0

            # 후보 풀
            pool = [c for c in courses if ["","사상/역사","사회/문화","문학/예술","과학/기술","건강/레포츠","외국어","융복합"][c["area"]] in areas]
            pool.sort(key=lambda x: -x["score"])

            # 랜덤 조합 1000번 시도
            found = False
            for _ in range(1000):
                picks = random.sample(pool[:40], min(len(pool), num))
                # 시간 겹침 체크
                all_slots = [s for c in picks for s in c["slots"]]
                overlap = False
                for i in range(len(all_slots)):
                    for j in range(i+1, len(all_slots)):
                        a, b = all_slots[i], all_slots[j]
                        if a["day"] == b["day"] and max(a["start"], b["start"]) < min(a["end"], b["end"]):
                            overlap = True
                if not overlap:
                    st.success("성공! 시간 겹침 없음!")
                    for c in picks:
                        tag = "AI 강추!" if keyword and c["score"] > 0.6 else ""
                        st.write(f"• **{c['name']}** ({c['prof']}) {tag}")
                    st.balloons()
                    found = True
                    break
            if not found:
                st.error("조건에 맞는 시간표를 못 찾았어요… 조금 더 유연하게 선택해보세요!")
