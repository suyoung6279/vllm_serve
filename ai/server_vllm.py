import json
import os
import re
import time
from datetime import date

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()

MODEL_NAME = os.getenv("MODEL_NAME", "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

client = OpenAI(base_url=VLLM_BASE_URL, api_key="dummy")


# ── 요청/응답 스키마 ───────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    text: str

class GenerateResponse(BaseModel):
    cotent: str
    todo_list: list

class AgendaRequest(BaseModel):
    title: str
    previous_summary: str = ""
    ocr_text: str = ""

class AgendaResponse(BaseModel):
    agendas: list

class ChatRequest(BaseModel):
    question: str
    context: str = ""
    history: list = []
    sources: list = []

class ChatResponse(BaseModel):
    answer: str
    citations: list = []


# ── 공통: LLM 호출 ─────────────────────────────────────────────────────────────
def call_llm(messages: list, max_tokens: int = 2048) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"repetition_penalty": 1.1},
    )
    raw = str(response.choices[0].message.content or "").strip()
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def parse_json(raw: str) -> dict:
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    raise ValueError(f"JSON 파싱 실패: {raw[:200]}")


# ── 회의록 요약 ────────────────────────────────────────────────────────────────
def extract_minutes(meeting_text: str) -> dict:
    prompt = f"""[역할]
    너는 회의를 화자별로 분석한 원본 데이터를 분석하여 회의록을 작성하고, 해야 할 일 즉 태스크를 추출하는 Jira AI야
    한눈에 읽어도 누구나 바로 알 수 있을 정도로 완벽한 정리를 해내지. 해당 내용들은 jira 이슈에도 등록될 예정이야 그러니
    매우 깔끔하고 정확하게 추출해야해 중국어 일본어 절대 쓰지마.

    [미션]
    제공된 대화 텍스트를 분석하여 반드시 아래 JSON 형식으로만 응답해줘.
    텍스트에 명시되지 않은 정보는 억지로 지어내지 말고 "미정" 또는 "없음"으로 처리해
    모든 내용은 반드시 100% 한국어(Korean)로만 작성해라. 중국어나 일본어는 절대 사용 금지.

    [담당자 추출 규칙 - 매우 중요]
    회의록은 "이름: 발언내용" 형식으로 구성되어 있어.
    담당자를 결정할 때 아래 규칙을 반드시 따라:

    패턴 1 - 본인이 직접 선언하는 경우 → 발언한 화자가 담당자
      예) "김규호: 저는 발표 자료 만들겠습니다" → assignee: 김규호
      예) "류지우: 저는 STT 테스트 해볼게요" → assignee: 류지우

    패턴 2 - 리더가 다른 사람에게 지시하는 경우 → 지시받은 사람이 담당자
      예) "김지원: 규호님은 체크리스트 부탁드립니다" → assignee: 김규호
      예) "김지원: 민준님이 공고 수정해주세요" → assignee: 김민준

    [담당자 추출 주의사항 - 반드시 지켜]
    - 회의 진행자(사회자)가 "다음 회의에서 논의하자", "검토해보자" 같은 말을 했다고 해서 진행자를 담당자로 지정하지 마.
    - 명확하게 특정인이 하겠다고 선언하거나, 특정인에게 지시한 경우에만 담당자를 지정해.
    - 담당자가 불분명하면 "미정"으로 처리해. 억지로 진행자 이름 넣지 마.
    - 한 사람에게 태스크가 몰리면 다시 검토해. 특히 진행자에게 태스크가 집중되는 건 잘못된 거야.

    [우선순위 기준 - 반드시 지켜]
    - High: 이번 주 안에 반드시 완료해야 하는 긴급한 업무
    - Medium: 중요하지만 약간의 여유가 있는 업무
    - Low: 중요도가 낮거나 나중에 해도 되는 업무
    - Lowest: 당장 급하지 않고 장기적으로 검토할 업무
    - "슬슬 정리해야 한다", "나중에 검토", "다음 회의에서" 같은 표현이 나오면 Low 또는 Lowest로 처리해.

    [출력 형식]
    {{
      "cotent": "회의록 내용을 정리해줘. 줄바꿈은 \\n으로 표현하고 JSON 문자열 안에서 실제 줄바꿈 금지. 형식: 큰 주제\\n1. 세부 내용\\n  - 더 구체적인 내용.",
      "todo_list": [
        {{
          "title": "해당 담당자가 해야 할 업무 내용(태스크 명)",
          "content": "해당 담당자가 해야 할 구체적인 업무 내용",
          "owner": "담당자 이름 (불분명하면 미정)",
          "due_date": "마감 일정(YYYY-MM-DD 형식, 오늘 날짜 기준으로 설정)",
          "priority": "우선 순위(High, Medium, Low, Lowest 중 하나)"
        }}
      ]
    }}

    [회의록 텍스트]
    {meeting_text}
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Always respond in valid JSON format only, with no extra text."},
        {"role": "user", "content": prompt},
    ]
    return parse_json(call_llm(messages))


# ── 기초 안건 생성 ──────────────────────────────────────────────────────────────
def generate_agendas_fn(req: AgendaRequest) -> dict:
    prompt = f"""회의 기초 안건을 생성해줘.

규칙:
- 회의 제목이 있으면 3~5개의 안건을 생성한다.
- 이전 회의 요약과 OCR 참고 텍스트가 있으면 반영한다.
- 입력에 없는 사람, 날짜, 회사명, 문서명은 지어내지 않는다.
- 반드시 한국어 JSON만 반환한다.

출력 형식:
{{"agendas": [{{"title": "안건명", "content": "논의할 내용"}}]}}

회의 제목: {req.title}
이전 회의 요약: {req.previous_summary or "없음"}
OCR 참고 텍스트: {req.ocr_text or "없음"}
"""
    messages = [
        {"role": "system", "content": "Return valid Korean JSON only. Do not return an empty agendas array when a title is provided."},
        {"role": "user", "content": prompt},
    ]
    return parse_json(call_llm(messages, max_tokens=1024))


# ── 챗봇 ───────────────────────────────────────────────────────────────────────
def chat_answer(req: ChatRequest) -> dict:
    history = "\n".join(f"{item.get('role','user')}: {item.get('content','')}" for item in req.history[-8:])
    prompt = f"""질문에 답해줘.

규칙:
- 반드시 [컨텍스트]에 있는 내용만 사용한다.
- 컨텍스트에 없으면 "제공된 자료에서 확인할 수 없습니다."라고 답한다.
- citations에는 실제로 사용한 출처만 sources에서 그대로 복사한다.
- JSON만 반환한다.

출력 형식:
{{"answer": "답변", "citations": ["출처"]}}

[대화 이력]
{history or "없음"}

[사용 가능한 출처]
{json.dumps(req.sources, ensure_ascii=False)}

[컨텍스트]
{req.context or "없음"}

[질문]
{req.question}
"""
    messages = [
        {"role": "system", "content": "You answer Korean questions using only the provided context. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    return parse_json(call_llm(messages, max_tokens=1024))


# ── 엔드포인트 ─────────────────────────────────────────────────────────────────
@app.get("/health")
@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate", response_model=GenerateResponse)
def generate_minutes(req: GenerateRequest):
    start = time.time()
    try:
        result = extract_minutes(req.text)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    print(f"✅ 회의록 생성 완료 | {time.time() - start:.1f}초")
    return result


@app.post("/generate-agendas", response_model=AgendaResponse)
def create_agendas(req: AgendaRequest):
    start = time.time()
    try:
        result = generate_agendas_fn(req)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    print(f"✅ 안건 생성 완료 | {time.time() - start:.1f}초")
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    start = time.time()
    try:
        result = chat_answer(req)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    print(f"✅ 챗봇 응답 완료 | {time.time() - start:.1f}초")
    return result
