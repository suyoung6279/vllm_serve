# RunPod Model I/O Spec

이 문서는 RunPod Core 서버의 세 모델 API 입출력 계약을 DB 저장/연동 기준으로 정리한 것이다.

기준 코드:
- `runpod/schemas.py`
- `runpod/core_server.py`
- `runpod/prompts.py`

## 공통 응답 형식

세 API 모두 최상위 응답은 `TextResponse` 형식이다.

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `result` | 필수 | object | 모델별 실제 결과 본문 |
| `elapsed_sec` | 필수 | number | RunPod 처리 시간(초) |
| `model` | 필수 | string | 사용한 텍스트 모델 ID |

DB에는 보통 `result` 내부 값을 저장하면 된다.

예시:

```json
{
  "result": {},
  "elapsed_sec": 12.345,
  "model": "unsloth/gemma-4-12B-it-qat-GGUF"
}
```

## 1. 기초안건 생성

Endpoint: `POST /generate-agendas`

Request model: `AgendaRequest`

### 입력

| 필드 | 필수 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `title` | 필수 | string | 없음 | 회의 제목. 안건 생성의 핵심 입력값 |
| `previous_summary` | 선택 | string | `""` | 이전 회의 요약 또는 참고 맥락 |
| `ocr_text` | 선택 | string | `""` | OCR로 추출한 참고 문서 텍스트 |

요청 예시:

```json
{
  "title": "회의록 자동화 서비스 개발 회의",
  "previous_summary": "지난 회의에서 STT와 OCR 연동 범위를 논의했다.",
  "ocr_text": "요구사항 문서 OCR 결과..."
}
```

### 출력

`result` 내부:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `agendas` | 필수 | array<object> | 생성된 기초 안건 목록 |

`agendas[]` 항목:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `title` | 필수 | string | 안건 제목 |
| `content` | 필수 | string | 안건 설명 또는 논의 내용 |

응답 예시:

```json
{
  "result": {
    "agendas": [
      {
        "title": "STT 회의 대본 저장 방식 확정",
        "content": "회의 대본을 어떤 테이블과 필드에 저장할지 결정한다."
      }
    ]
  },
  "elapsed_sec": 3.21,
  "model": "..."
}
```

DB 저장 추천:
- 안건 테이블: `agendas[].title`, `agendas[].content`
- 회의와 연결할 경우: 요청의 회의 ID는 이 API 스키마에는 없으므로 백엔드 라우터에서 별도로 매핑해야 한다.

## 2. 회의록 생성

Endpoint: `POST /generate-minutes`

Request model: `MinutesRequest`

### 입력

| 필드 | 필수 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `text` | 필수 | string | 없음 | STT 회의 대본 전체 텍스트. 현재 `req.text` |
| `meeting_id` | 선택 | string | `""` | 회의 ID. Qdrant 회의록 저장 메타데이터에도 사용 |
| `title` | 선택 | string | `""` | 회의 제목 |
| `meeting_datetime` | 선택 | string | `""` | 회의 일시 |
| `location` | 선택 | string | `""` | 회의 장소 |

요청 예시:

```json
{
  "text": "김민수: 오늘은 STT 연동부터 확인하겠습니다.\n박지현: 제가 API 명세를 정리하겠습니다.",
  "meeting_id": "42",
  "title": "회의록 자동화 서비스 개발 회의",
  "meeting_datetime": "2026-06-08T10:00:00",
  "location": "A 회의실"
}
```

### 출력

`result` 내부:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `summary` | 필수 | string | 회의 핵심 요약 |
| `cotent` | 필수 | string | 회의록 본문. 현재 프롬프트의 키가 `cotent`로 되어 있음 |
| `todo_list` | 필수 | array<object> | 회의에서 추출한 업무 목록 |
| `qdrant_ingest` | 선택 | object | 회의록을 Qdrant에 저장한 결과 |
| `qdrant_ingest_error` | 선택 | string | Qdrant 저장 실패 시 에러 문자열 |

주의:
- 현재 백엔드 저장 코드는 `result.cotent`를 우선 사용하고, 없으면 `result.content`도 fallback으로 읽는다.
- DB 필드명을 새로 정한다면 `content`가 더 자연스럽지만, 현재 RunPod 프롬프트 기준 응답 키는 `cotent`다.

`todo_list[]` 항목:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `title` | 필수 | string | 업무 제목 |
| `content` | 필수 | string | 업무 상세 내용 |
| `owner` | 필수 | string | 담당자 이름 |
| `due_date` | 필수 | string | `YYYY-MM-DD` 또는 `미정` |
| `priority` | 필수 | string | `High`, `Medium`, `Low`, `Lowest` 중 하나 |

응답 예시:

```json
{
  "result": {
    "summary": "STT 대본 저장 방식과 회의록 생성 API 연동 범위를 논의했다.",
    "cotent": "회의록 본문...",
    "todo_list": [
      {
        "title": "API 입출력 명세 정리",
        "content": "기초안건, 회의록, 준비자료 생성 API의 입력과 출력을 정리한다.",
        "owner": "박지현",
        "due_date": "2026-06-10",
        "priority": "High"
      }
    ],
    "qdrant_ingest": {
      "source_type": "meeting_minutes",
      "upserted": 3
    }
  },
  "elapsed_sec": 15.2,
  "model": "..."
}
```

DB 저장 추천:
- 회의 요약: `result.summary`
- 회의록 본문: `result.cotent` 또는 `result.content`
- 업무/태스크: `result.todo_list[]`
- Qdrant 상태 로그: `result.qdrant_ingest`, `result.qdrant_ingest_error`

## 3. 준비자료 생성

Endpoint: `POST /generate-preparation`

Request model: `PreparationRequest`

### 입력

| 필드 | 필수 | 타입 | 기본값 | 설명 |
| --- | --- | --- | --- | --- |
| `title` | 필수 | string | 없음 | 회의 제목 또는 준비자료 생성 주제 |
| `participants` | 선택 | array<object> | `[]` | 참석자 정보 |
| `agendas` | 선택 | array<string> | `[]` | 선택된 안건 목록 |
| `previous_meetings` | 선택 | array<object> | `[]` | 백엔드에서 직접 넘기는 이전 회의 참고자료 |
| `internal_documents` | 선택 | array<object> | `[]` | 백엔드에서 직접 넘기는 내부 문서 참고자료 |

참고:
- `previous_meetings` 또는 `internal_documents`가 비어 있으면 서버가 Qdrant에서 관련 문서를 찾는다.
- 외부 뉴스는 요청 body로 받지 않고 서버가 `title`/검색 query 기반으로 조회한다.
- 현재 `PreparationRequest`에는 `meeting_id`, `meeting_datetime`, `location` 필드가 정의되어 있지 않다.

`participants[]` 권장 항목:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `name` | 선택 | string | 참석자 이름 |
| `work` | 선택 | string | 직무 또는 역할 |

`previous_meetings[]`, `internal_documents[]` 권장 항목:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `source` | 선택 | string | 출처명 |
| `title` | 선택 | string | 문서 제목. `source`가 없으면 source 대체값으로 사용 가능 |
| `text` | 선택 | string | 문서 본문 |
| `content` | 선택 | string | `text`가 없을 때 본문 대체값 |
| `category` | 선택 | string | 문서 분류 |
| `source_type` | 선택 | string | 출처 유형 |

요청 예시:

```json
{
  "title": "회의록 자동화 서비스 개발 회의",
  "participants": [
    { "name": "김민수", "work": "백엔드" },
    { "name": "박지현", "work": "프론트엔드" }
  ],
  "agendas": [
    "STT 회의 대본 저장 방식 확정",
    "Qdrant 기반 챗봇 검색 품질 점검"
  ],
  "previous_meetings": [],
  "internal_documents": []
}
```

### 출력

`result` 내부:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `document` | 필수 | string | 최종 준비자료 문서. DB 저장 1순위 |
| `sections` | 필수 | array<object> | 준비자료 섹션 목록 |
| `selected_documents` | 필수 | object | 실제 사용/선택된 참고자료 목록 |
| `source_map` | 필수 | array<object> | 출처와 사용 이유 매핑 |
| `retrieval` | 선택 | object | 서버가 추가하는 검색 상태/개수 정보 |
| `fallback_error` | 선택 | string | 모델 생성 실패로 fallback 문서를 만든 경우 에러 |

`sections[]` 항목:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `title` | 필수 | string | 섹션 제목 |
| `content` | 필수 | string | 섹션 내용 |
| `sources` | 선택 | array<string> | 섹션에 사용한 출처 |

`selected_documents` 항목:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `previous_meetings` | 필수 | array<object> | 선택된 이전 회의 자료 |
| `internal_documents` | 필수 | array<object> | 선택된 내부 문서 |
| `external_news` | 필수 | array<object> | 검색된 외부 뉴스 |

`retrieval` 항목:

| 필드 | 필수 | 타입 | 설명 |
| --- | --- | --- | --- |
| `query` | 필수 | string | 검색에 사용한 query |
| `qdrant_collection` | 필수 | string | 검색한 Qdrant collection. 검색 실패 시 빈 문자열 |
| `previous_count` | 필수 | number | 이전 회의 자료 개수 |
| `internal_count` | 필수 | number | 내부 문서 개수 |
| `news_count` | 필수 | number | 외부 뉴스 개수 |
| `retrieval_error` | 필수 | string | Qdrant 검색 오류. 정상일 때 빈 문자열 |
| `news_error` | 필수 | string | 뉴스 검색 오류. 정상일 때 빈 문자열 |

응답 예시:

```json
{
  "result": {
    "document": "### 회의 준비자료\n\n#### 1. 회의 주제\n- 회의록 자동화 서비스 개발 회의",
    "sections": [
      {
        "title": "회의 주제",
        "content": "회의록 자동화 서비스 개발 회의",
        "sources": []
      }
    ],
    "selected_documents": {
      "previous_meetings": [],
      "internal_documents": [],
      "external_news": []
    },
    "source_map": [],
    "retrieval": {
      "query": "회의록 자동화 서비스 개발 회의",
      "qdrant_collection": "hpm_documents",
      "previous_count": 0,
      "internal_count": 0,
      "news_count": 0,
      "retrieval_error": "",
      "news_error": ""
    }
  },
  "elapsed_sec": 8.4,
  "model": "..."
}
```

DB 저장 추천:
- 준비자료 본문: `result.document`
- 문서가 비어 있을 때 fallback: `sections[].title + sections[].content`를 합쳐 저장
- 참고자료 로그: `result.selected_documents`
- 검색 상태 로그: `result.retrieval`
- 출처 매핑: `result.source_map`

## 빠른 매핑 요약

| 모델 | Endpoint | 입력 필수 | 주요 출력 |
| --- | --- | --- | --- |
| 기초안건 생성 | `/generate-agendas` | `title` | `result.agendas[]` |
| 회의록 생성 | `/generate-minutes` | `text` | `result.summary`, `result.cotent`, `result.todo_list[]` |
| 준비자료 생성 | `/generate-preparation` | `title` | `result.document`, `result.sections[]`, `result.selected_documents`, `result.retrieval` |

