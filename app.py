import streamlit as st
import pandas as pd
import os
import re
import random
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────
# 1. Streamlit Cloud에서 CSV 강제 로드 (이거 없으면 안 읽힘!)
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
st.title("🧠 2025-2학기 AI 시간표 생성기")
st.markdown("**교수명 O | 시간 겹침 0% | '운동' '경제' '영어'만 써도 AI가 알아서 추천**")

# ─────────────────────────────────────
# 3. AI 모델 로딩
# ─────────────────────────────────────
@st.cache_resource
def load_model():
    with st.spinner("AI 모델 로딩 중… (최초 20~40초)"):
        return SentenceTransformer('jhgan/ko-sroberta-multitask')
model = load_model()

# ─────────────────────────────────────
# 4. 완전 강력한 시간 파싱 (모든 형식 다 잡음!)
# ─────────────────────────────────────
def parse_time(text):
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r'<br\s*/?>|\n', ' ', text)
    slots = []
    yoil_map = {"월":0, "화":1, "수":2, "목":3, "금":4}
    # 패턴 1: 월 09:00-10:30, 월 9:00~10:30
    p1 = re.finditer(r"([월화수목금])\s*(\d{1,2}:\d{2})\s*[-~]\s*(\d{1,2}:\d{2})", text)
    # 패턴 2: 월 09:00(90)
    p2 = re.finditer(r"([월화수목금])\s*(\d{1,2}:\d{2})\s*\(\s*(\d+)\s*\)", text)
    
    for m in list(p1) + list(p2):
        day_str = m.group(1)
        start_str = m.group(2)
        if len(m.groups()) == 3 and m.group(3).isdigit():  # (90) 형식
            duration = int(m.group(3))
            h, mn = map(int, start_str.split(":"))
            start_min = h*60 + mn
            end_min = start_min + duration
            end_str = f"{end_min//60}:{end_min%60:02d}"
        else:
            end_str = m.group(3)
        try:
            day = yoil_map[day_str]
            sh, sm = map(int, start_str.split(":"))
            eh, em = map(int, end_str.split(":"))
            start = sh*60 + sm
            end = eh*60 + em
            if end > start:
                slots.append({"day": day, "start": start, "end": end})
        except:
            continue
    return slots

# ─────────────────────────────────────
# 5. 데이터 로드
# ─────────────────────────────────────
courses = []
total = 0
for i in range(1, 8):
    file = f"section{i}.csv"
    if os.path.exists(file):
        df = pd.read_csv(file, encoding="cp949").fillna("")
        total += len(df)
        for _, row in df.iterrows():
            name = str(row.get("교과목명(미확정구분)", "")).strip()
            prof = str(row.get("교수명", "미정")).strip()
            time = str(row.get("시간/강의실", ""))
            slots = parse_time(time)
            if name and slots:
                courses.append({
                    "name": name,
                    "prof": prof,
                    "area": i,
                    "slots": slots,
                    "search": f"{name} {prof} {['','사상/역사','사회/문화','문학/예술','과학/기술','건강/레포츠','외국어','융복합'][i]}"
                })

if not courses:
    st.error("CSV 파일을 읽지 못했습니다. 파일 이름 확인!")
    st.stop()

st.success(f"총 {len(courses)}개 교양 과목 로드 완료! (전체 {total}개 중 시간표 있는 과목)")

# 임베딩
embeddings = model.encode([c["search"] for c in courses], convert_to_tensor=True)

# ─────────────────────────────────────
# 6. UI & 생성 로직
# ─────────────────────────────────────
col1, col2 = st.columns([1,1])
with col1:
    st.subheader("영역 선택")
    area_names = ["사상/역사","사회/문화","문학/예술","과학/기술","건강/레포츠","외국어","융복합"]
    selected_areas = st.multiselect("복수 선택 가능", area_names, default=["건강/레포츠","외국어"])

with col2:
    st.subheader("설정")
    num = st.selectbox("교양 과목 수", [1,2,3], 1)
    keyword = st.text_input("AI 검색 (선택)", placeholder="예: 운동, 경제, 영어, 철학, 코딩")

if st.button("시간표 생성 🚀", type="primary"):
    if not selected_areas:
        st.error("영역을 하나 이상 선택해주세요!")
    else:
        with st.spinner("AI가 최고의 시간표 찾는 중…"):
            # 점수 계산
            if keyword:
                sims = util.cos_sim(model.encode(keyword), embeddings)[0].cpu().numpy()
                for i, s in enumerate(sims):
                    courses[i]["score"] = float(s) * 100 + (30 if keyword in courses[i]["name"] else 0)
            else:
                for c in courses: c["score"] = c.get("rating", 0)

            # 후보 풀
            pool = [c for c in courses if area_names[c["area"]-1] in selected_areas]
            pool.sort(key=lambda x: -x["score"])

            results = []
            for _ in range(3000):
                picks = random.sample(pool[:70], min(len(pool), num))
                # 시간 겹침 체크
                all_slots = [s for c in picks for s in c["slots"]]
                overlap = False
                for i in range(len(all_slots)):
                    for j in range(i+1, len(all_slots)):
                        a, b = all_slots[i], all_slots[j]
                        if a["day"] == b["day"] and max(a["start"], b["start"]) < min(a["end"], b["end"]):
                            overlap = True
                            break
                    if overlap: break
                if not overlap:
                    score = sum(c["score"] for c in picks)
                    results.append({"score": score, "picks": picks})
                    if len(results) >= 5: break

            if results:
                results.sort(key=lambda x: -x["score"])
                for idx, r in enumerate(results[:3]):
                    with st.expander(f"추천 {idx+1}위 (점수: {r['score']:.1f})"):
                        for c in r["picks"]:
                            tag = "✨ AI 추천" if keyword and c["score"] > 50 else ""
                            st.write(f"• {c['name']} ({c['prof']}) {tag}")
                        st.success("시간 겹침 없음!")
                        st.balloons()
            else:
                st.error("조건에 맞는 시간표를 찾지 못했습니다. 영역을 늘리거나 키워드를 바꿔보세요!")
