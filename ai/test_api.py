import requests
import json

# ── 설정 ──────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8501"  # RunPod 프록시 주소로 교체
# BASE_URL = "https://xxxxx-8501.proxy.runpod.net"


def print_result(title: str, response: requests.Response):
    print(f"\n{'='*60}")
    print(f"[{title}]")
    print(f"상태코드: {response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(response.text)
    print(f"{'='*60}")


# ── 1. 헬스 체크 ───────────────────────────────────────────────────────────────
def test_health():
    res = requests.get(f"{BASE_URL}/health")
    print_result("헬스 체크", res)


# ── 2. 회의록 요약 ──────────────────────────────────────────────────────────────
def test_generate_minutes():
    payload = {
        "text": """
김지원: 오늘 회의 시작하겠습니다. 이번 주 목표는 STT 모듈 완성이에요.
류지우: 저는 WhisperX 테스트 완료했고, 정확도 92% 나왔습니다.
김규호: 저는 발표 자료 만들겠습니다. 금요일까지 완성할게요.
김지원: 박수영님은 UI 디자인 부탁드립니다.
박수영: 네, 다음 주 월요일까지 완성할게요.
김지원: 좋습니다. 다음 회의는 금요일 오후 3시입니다.
        """.strip(),
        "title": "STT 모듈 개발 회의",
        "meeting_datetime": "2026-06-09 14:00",
    }
    res = requests.post(f"{BASE_URL}/generate", json=payload)
    print_result("회의록 요약", res)


# ── 3. 기초 안건 생성 ──────────────────────────────────────────────────────────
def test_generate_agendas():
    payload = {
        "title": "2분기 마케팅 전략 회의",
        "previous_summary": "1분기에는 SNS 광고 중심으로 진행했으며 전환율 3.2% 달성",
        "ocr_text": "",
    }
    res = requests.post(f"{BASE_URL}/generate-agendas", json=payload)
    print_result("기초 안건 생성", res)


# ── 4. 챗봇 ───────────────────────────────────────────────────────────────────
def test_chat():
    payload = {
        "question": "STT 모듈 담당자가 누구야?",
        "context": """
2026-06-09 STT 모듈 개발 회의:
- 류지우: WhisperX 테스트 완료, 정확도 92%
- 김규호: 발표 자료 담당, 금요일 완성 예정
- 박수영: UI 디자인 담당, 다음 주 월요일 완성 예정
        """.strip(),
        "history": [],
        "sources": ["STT 모듈 개발 회의 (2026-06-09)"],
    }
    res = requests.post(f"{BASE_URL}/chat", json=payload)
    print_result("챗봇", res)


# ── 실행 ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"테스트 대상: {BASE_URL}")

    test_health()
    test_generate_minutes()
    test_generate_agendas()
    test_chat()

    print("\n✅ 전체 테스트 완료")
