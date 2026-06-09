from __future__ import annotations

import json
from datetime import date
from typing import Any

from schemas import AgendaRequest, ChatRequest, MinutesRequest, PreparationRequest


def minutes_messages(req: MinutesRequest) -> list[dict[str, str]]:
    today = date.today().isoformat()
    prompt = f"""
[역할]
너는 회의를 화자별로 분석한 원본 데이터를 분석하여 회의록을 작성하고, 해야 할 일 즉 태스크를 추출하는 Jira AI야.
한눈에 읽어도 누구나 바로 알 수 있을 정도로 완벽한 정리를 해내지. 해당 내용들은 Jira 이슈에도 등록될 예정이야.
그러니 매우 깔끔하고 정확하게 추출해야 해. 중국어 일본어 절대 쓰지마.

[미션]
제공된 대화 텍스트를 분석하여 반드시 아래 JSON 형식으로만 응답해줘.
텍스트에 명시되지 않은 정보는 억지로 지어내지 말고 "미정" 또는 "없음"으로 처리해.
모든 내용은 반드시 100% 한국어(Korean)로만 작성해라. 중국어나 일본어는 절대 사용 금지.

[담당자 추출 규칙 - 매우 중요]
회의록은 "이름: 발언내용" 형식으로 구성되어 있어.
담당자를 결정할 때 아래 두 가지 패턴을 반드시 구분해:

패턴 1 - 본인이 직접 선언하는 경우 → 발언한 화자가 담당자
  예) "김규호: 저는 발표 자료 만들겠습니다" → assignee: 김규호
  예) "류지우: 저는 STT 테스트 해볼게요" → assignee: 류지우
  예) "박수영: 제가 디자인 맡을게요" → assignee: 박수영

패턴 2 - 리더가 다른 사람에게 지시하는 경우 → 지시받은 사람이 담당자
  예) "김지원: 규호님은 체크리스트 부탁드립니다" → assignee: 김규호
  예) "김지원: 민준님이 공고 수정해주세요" → assignee: 김민준

[날짜 기준]
- 오늘 날짜: {today}
- due_date는 회의 텍스트에 명확한 날짜가 있으면 그 날짜를 사용한다.
- "이번 주", "다음 주", "금요일까지"처럼 상대 일정이 있으면 오늘 날짜를 기준으로 YYYY-MM-DD 형식으로 계산한다.
- 마감 일정이 전혀 없으면 "미정"으로 둔다.

[회의 메타데이터]
- 회의 ID: {req.meeting_id or "미정"}
- 회의 제목: {req.title or "미정"}
- 회의 일시: {req.meeting_datetime or "미정"}
- 장소: {req.location or "미정"}

[출력 형식]
{{
  "summary": "회의 핵심 내용을 1~2문장으로 요약",
  "cotent": "회의록내용을 정리해줘. 누가 봐도 한눈에 이해할 정도로 써야해. 너무 간략하게 쓰면 이해하기 어렵다. 줄바꿈은 \\n으로 표현하고 JSON 문자열 안에서 실제 줄바꿈 금지. 형식: 큰 주제\\n1. 세부 내용\\n  - 더 구체적인 내용.",
  "todo_list": [
    {{
      "title": "해당 담당자가 해야 할 업무 내용(태스크 명)",
      "content": "해당 담당자가 해야 할 구체적인 업무 내용",
      "owner": "담당자 이름",
      "due_date": "마감 일정(YYYY-MM-DD 또는 미정)",
      "priority": "High|Medium|Low|Lowest"
    }}
  ]
}}

[프로젝트 누적 맥락]
{req.project_context or "없음"}

[회의 대본]
{req.text}
""".strip()
    return [
        {"role": "system", "content": "You are a helpful assistant. Always respond in valid JSON format only, with no extra text."},
        {"role": "user", "content": prompt},
    ]


def agenda_messages(req: AgendaRequest, *, retry: bool = False) -> list[dict[str, str]]:
    if retry:
        prompt = f'회의 제목 "{req.title}"만 근거로 회의 기초 안건 4개를 생성해줘. 출력은 {{"agendas":[{{"title":"...","content":"..."}}]}} JSON만 반환해.'
    else:
        prompt = f"""
회의 기초 안건을 생성해줘.

규칙:
- 회의 제목이 있으면 3~5개의 안건을 생성한다.
- 이전 회의 요약과 OCR 참고 텍스트가 있으면 반영한다.
- 입력에 없는 사람, 날짜, 회사명, 문서명은 지어내지 않는다.
- 반드시 한국어 JSON만 반환한다.

출력 형식:
{{"agendas": [{{"title": "안건명", "content": "논의할 내용"}}]}}

회의 제목: {req.title}
이전 회의 요약: {req.previous_summary}
OCR 참고 텍스트: {req.ocr_text}
""".strip()
    return [
        {"role": "system", "content": "Return valid Korean JSON only. Do not return an empty agendas array when a title is provided."},
        {"role": "user", "content": prompt},
    ]


def preparation_messages(req: PreparationRequest, selected_documents: dict[str, Any]) -> list[dict[str, str]]:
    prompt = f"""
당신은 프로젝트 문서 분석 및 회의 준비 자료 작성 전문가입니다.
제공된 문서를 바탕으로 객관적인 '회의 준비 자료'를 작성하십시오.

[핵심 규칙]
1. 제공된 자료([프로젝트 히스토리], [내부 문서], [외부 뉴스])에 명시된 내용만 사용합니다. 사실을 추정하거나 지어내지 마십시오.
2. 문서에서 확인되지 않거나 누락된 정보는 반드시 "확인 필요"로 표기합니다.
3. 모든 항목의 끝에는 정보의 출처를 [문서명] 형태로 표기합니다. (프로젝트 히스토리/회의 주제는 생략 가능)
4. 주어진 [회의 주제]와 직접 관련 없는 정보는 제외합니다.
5. 출력은 반드시 유효한 JSON 객체 하나만 반환합니다. 마크다운 본문은 document 필드 안에 JSON 문자열로 작성합니다.
6. JSON 밖에는 코드블록, 설명 문장, 마크다운을 절대 출력하지 마십시오.
7. 최상위 key는 반드시 "document", "sections", "selected_documents", "source_map" 네 개만 사용합니다.

[입력 데이터]
- 회의 주제: {req.title}
- 프로젝트 누적 맥락: {req.project_context or "없음"}
- 참석자: {json.dumps(req.participants, ensure_ascii=False)}
- 안건: {json.dumps(req.agendas, ensure_ascii=False)}
- 프로젝트 히스토리: {json.dumps(selected_documents.get("previous_meetings", []), ensure_ascii=False)}
- 내부 문서: {json.dumps(selected_documents.get("internal_documents", []), ensure_ascii=False)}
- 외부 뉴스: {json.dumps(selected_documents.get("external_news", []), ensure_ascii=False)}

[출력 형식]
반드시 아래 JSON 스키마만 반환하십시오.

{{
  "document": "### 회의 준비 자료\\n\\n#### 1. 회의 주제\\n- {req.title}\\n\\n#### 2. 회의 목적\\n- ...\\n\\n#### 3. 프로젝트 현재 상태\\n- **완료**: ...\\n- **진행 중**: ...\\n\\n#### 4. 관련 규정 및 제약사항\\n- ... [출처]\\n\\n#### 5. 회의 종료 후 기대 결과\\n- ...\\n\\n#### 6. 참조 문서 목록\\n- ...",
  "sections": [
    {{
      "title": "회의 주제",
      "content": "{req.title}",
      "sources": []
    }}
  ],
  "selected_documents": {{
    "previous_meetings": [],
    "internal_documents": [],
    "external_news": []
  }},
  "source_map": [
    {{
      "source": "문서명 또는 source 값",
      "used_for": "사용 이유"
    }}
  ]
}}
""".strip()
    return [
        {
            "role": "system",
            "content": "Return exactly one valid JSON object and nothing else. "
            "Do not use markdown outside JSON. "
            "The JSON object must have exactly these top-level keys: "
            "document, sections, selected_documents, source_map. "
            "The document field must contain Korean markdown as a JSON string. "
            "Use only the provided evidence. If evidence is missing, write '확인 필요'. "
            "Do not invent facts.",
        },
        {"role": "user", "content": prompt},
    ]


def legacy_preparation_messages(req: PreparationRequest) -> list[dict[str, str]]:
    selected = {
        "previous_meetings": [],
        "internal_documents": [],
        "external_news": [],
    }
    return preparation_messages(req, selected)


def chat_messages(req: ChatRequest, context: str, sources: list[str]) -> list[dict[str, str]]:
    history = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in req.history[-8:])
    prompt = f"""
질문에 답해줘.

규칙:
- 반드시 [Qdrant 컨텍스트]에 있는 내용만 사용한다.
- 컨텍스트에 없으면 "제공된 자료에서 확인할 수 없습니다."라고 답한다.
- citations에는 실제로 사용한 출처만 sources에서 그대로 복사한다.
- JSON만 반환한다.

출력 형식:
{{"answer": "답변", "citations": ["출처"]}}

[대화 이력]
{history}

[사용 가능한 출처]
{json.dumps(sources, ensure_ascii=False)}

[Qdrant 컨텍스트]
{context}

[질문]
{req.question}
""".strip()
    return [
        {"role": "system", "content": "You answer Korean meeting questions using only the provided Qdrant context. Return valid JSON only."},
        {"role": "user", "content": prompt},
    ]
