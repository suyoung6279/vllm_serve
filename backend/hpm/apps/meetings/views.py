import os
import requests
from datetime import datetime
from django.conf import settings
from django.shortcuts import render
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response

from apps.documents.models import Document
from apps.projects.models import Project, ProjectUsers
from .models import Meeting, MeetingAgendas, MeetingPreparation, MeetingTask, MeetingUsers, Record
from .serializers import MeetingSerializer

# RunPod이 반환하는 priority 문자열 → DB 정수 변환
PRIORITY_MAP = {
    "High": 1,
    "Medium": 2,
    "Low": 3,
    "Lowest": 4,
}

PRIORITY_LABELS = {value: key for key, value in PRIORITY_MAP.items()}


def index(request):
    return render(request, "base.html")


def runpod_url(path: str, service: str = "core") -> str:
    base_url = {
        "core": settings.RUNPOD_CORE_BASE_URL,
        "stt": settings.RUNPOD_STT_BASE_URL,
        "ocr": settings.RUNPOD_OCR_BASE_URL,
    }.get(service, settings.RUNPOD_BASE_URL)
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def runpod_post(path: str, payload: dict, timeout: int = 600) -> Response:
    try:
        response = requests.post(runpod_url(path), json=payload, timeout=timeout)
        response.raise_for_status()
        return Response(response.json(), status=status.HTTP_200_OK)
    except requests.RequestException as e:
        return Response({"error": f"RunPod 연결 실패: {str(e)}"}, status=status.HTTP_502_BAD_GATEWAY)


def runpod_transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as audio:
        response = requests.post(
            runpod_url("/stt", "stt"),
            files={"file": (os.path.basename(file_path), audio, "audio/webm")},
            data={"language": "ko", "align": "true", "diarize": "false"},
            timeout=1800,
        )
    response.raise_for_status()
    data = response.json()
    result = data.get("result", data) if isinstance(data, dict) else {}
    if isinstance(result, dict):
        text = result.get("text") or result.get("transcript") or result.get("record_row_text") or ""
        if text:
            return str(text).strip()
        segments = result.get("segments")
        if isinstance(segments, list):
            return "\n".join(
                str(segment.get("text") or "").strip()
                for segment in segments
                if isinstance(segment, dict) and str(segment.get("text") or "").strip()
            ).strip()
    return ""


def parse_meeting_at(value: str):
    parsed = parse_datetime(str(value or "").strip())
    if parsed is None:
        try:
            parsed = datetime.strptime(str(value).strip(), "%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            parsed = None
    if parsed is None:
        return None
    if timezone.is_naive(parsed):
        parsed = timezone.make_aware(parsed, timezone.get_current_timezone())
    return parsed


def previous_summary_text(project_id: int, exclude_meeting_id: int | None = None, limit: int = 3) -> str:
    queryset = Meeting.objects.filter(project_id=project_id).exclude(meeting_summary__isnull=True).exclude(meeting_summary="")
    if exclude_meeting_id:
        queryset = queryset.exclude(meeting_id=exclude_meeting_id)
    summaries = queryset.order_by("-meeting_at").values_list("title", "meeting_summary")[:limit]
    return "\n".join(f"- {title}: {summary}" for title, summary in summaries)


def append_project_context(project: Project, meeting: Meeting, summary: str, *, max_chars: int = 12000) -> None:
    summary = str(summary or "").strip()
    if not summary:
        return
    meeting_label = meeting.meeting_at.strftime("%Y-%m-%d") if meeting.meeting_at else "날짜 미정"
    entry = f"[{meeting_label}] {meeting.title}\n{summary}"
    current = str(project.context_summary or "").strip()
    if entry in current:
        return
    updated = f"{current}\n\n{entry}".strip() if current else entry
    if len(updated) > max_chars:
        updated = updated[-max_chars:].lstrip()
    project.context_summary = updated
    project.save(update_fields=["context_summary"])


def previous_meeting_sources(meeting: Meeting, limit: int = 5) -> list[dict]:
    meetings = (
        Meeting.objects.filter(project=meeting.project)
        .exclude(meeting_id=meeting.meeting_id)
        .exclude(meeting_summary__isnull=True)
        .exclude(meeting_summary="")
        .order_by("-meeting_at")[:limit]
    )
    return [
        {
            "source": f"[이전회의록: {item.title}][일시: {item.meeting_at:%Y-%m-%d %H:%M}]",
            "text": item.meeting_summary or item.meeting_document or "",
            "category": "previous_meeting",
        }
        for item in meetings
    ]


def internal_document_sources(meeting: Meeting, limit: int = 5) -> list[dict]:
    docs = Document.objects.filter(project=meeting.project).order_by("-uploaded_at")[:limit]
    sources = [
        {
            "source": f"[내부자료: {doc.title}]",
            "text": f"{doc.title}\n파일 경로: {doc.path}",
            "category": "internal_document",
        }
        for doc in docs
    ]
    if sources:
        return sources
    return [
        {
            "source": "[내부자료 더미: 회의 자동화 요구사항]",
            "text": "회의 생성 시 주제, 장소, 일시, 참여자를 저장하고 안건, 준비자료, 회의록, 업무를 자동 생성한다.",
            "category": "internal_document",
        },
        {
            "source": "[내부자료 더미: 업무 매핑 규칙]",
            "text": "Task 담당자는 회의 참여자 이름과 직무를 우선 기준으로 매핑하며 불명확한 경우 미정 처리한다.",
            "category": "internal_document",
        },
    ]


def local_meeting_chat_answer(meeting: Meeting, question: str) -> dict:
    participants = [
        {"name": item.user.name, "work": item.user.work}
        for item in MeetingUsers.objects.filter(meeting=meeting).select_related("user").order_by("meeting_users_id")
    ]
    agendas = list(MeetingAgendas.objects.filter(meeting=meeting).values_list("content", flat=True))
    preparation = MeetingPreparation.objects.filter(meeting=meeting).last()
    record = Record.objects.filter(meeting=meeting).last()

    question_text = question.lower()
    sources = [f"[회의: {meeting.title}]"]
    if preparation:
        sources.append("[준비자료]")
    if record and record.record_row_text:
        sources.append("[녹음/STT 원문]")

    if "참여" in question_text or "참석" in question_text or "누구" in question_text:
        people = ", ".join(f"{item['name']}({item['work']})" for item in participants) or "등록된 참여자가 없습니다"
        answer = f"이 회의의 참여자는 {people}입니다."
    elif "안건" in question_text:
        agenda_text = "\n".join(f"- {item}" for item in agendas) or "등록된 기초안건이 없습니다."
        answer = f"현재 등록된 기초안건은 다음과 같습니다.\n{agenda_text}"
    elif "준비" in question_text or "자료" in question_text:
        document = preparation.document if preparation else "아직 준비자료가 생성되지 않았습니다."
        answer = document[:1200]
    elif "주제" in question_text or "회의" in question_text:
        answer = f"회의 주제는 '{meeting.title}'이고, 장소는 '{meeting.location}', 일시는 {meeting.meeting_at:%Y-%m-%d %H:%M}입니다."
    else:
        prep_text = preparation.document[:700] if preparation else ""
        agenda_text = ", ".join(agendas[:3])
        answer = (
            f"현재 회의 '{meeting.title}' 기준으로 답변합니다. "
            f"주요 안건은 {agenda_text or '아직 없습니다'}. "
            f"{prep_text}"
        ).strip()

    return {
        "answer": answer,
        "citations": sources,
        "fallback": True,
        "fallback_reason": "Core RunPod chat endpoint is unavailable.",
    }


def serialize_meeting_detail(meeting: Meeting) -> dict:
    serializer = MeetingSerializer(meeting)
    record = Record.objects.filter(meeting=meeting).last()
    participants = MeetingUsers.objects.filter(meeting=meeting).select_related("user").order_by("meeting_users_id")
    project_user_ids = {
        item.user_id: item.project_users_id
        for item in ProjectUsers.objects.filter(project=meeting.project, user_id__in=[p.user_id for p in participants])
    }
    tasks = MeetingTask.objects.filter(meeting=meeting).select_related("meeting_users__user").order_by("meeting_task_id")
    agendas = MeetingAgendas.objects.filter(meeting=meeting).order_by("id")
    preparation = MeetingPreparation.objects.filter(meeting=meeting).last()
    data = dict(serializer.data)
    data.update(
        {
            "record": {
                "record_id": record.record_id,
                "record_path": record.record_path,
                "record_row_text": record.record_row_text,
            }
            if record
            else None,
            "participants": [
                {
                    "project_users_id": project_user_ids.get(item.user_id),
                    "meeting_users_id": item.meeting_users_id,
                    "user_id": item.user.users_id,
                    "name": item.user.name,
                    "work": item.user.work,
                }
                for item in participants
            ],
            "agendas": [item.content for item in agendas],
            "preparation": {
                "id": preparation.id,
                "document": preparation.document,
            }
            if preparation
            else None,
            "tasks": [
                {
                    "meeting_task_id": task.meeting_task_id,
                    "title": task.title,
                    "content": task.content,
                    "owner": task.meeting_users.user.name,
                    "due_date": task.due_date.isoformat() if task.due_date else "",
                    "priority": PRIORITY_LABELS.get(task.priority, "Medium"),
                    "status": task.status,
                }
                for task in tasks
            ],
        }
    )
    return data


@api_view(["GET"])
def app_config(request):
    return Response(
        {
            "runpod_base_url": settings.RUNPOD_BASE_URL,
            "runpod_core_base_url": settings.RUNPOD_CORE_BASE_URL,
            "runpod_stt_base_url": settings.RUNPOD_STT_BASE_URL,
            "runpod_ocr_base_url": settings.RUNPOD_OCR_BASE_URL,
            "runpod_minutes_url": settings.RUNPOD_MINUTES_URL,
        }
    )


@api_view(["GET"])
def runpod_health(request):
    try:
        response = requests.get(runpod_url("/health", "core"), timeout=30)
        response.raise_for_status()
        return Response(response.json(), status=status.HTTP_200_OK)
    except requests.RequestException as e:
        return Response({"error": f"RunPod 연결 실패: {str(e)}"}, status=status.HTTP_502_BAD_GATEWAY)


@api_view(["POST"])
def proxy_generate_agendas(request):
    return runpod_post(
        "/generate-agendas",
        {
            "title": request.data.get("title", ""),
            "previous_summary": request.data.get("previous_summary", ""),
            "ocr_text": request.data.get("ocr_text", ""),
        },
        timeout=300,
    )


@api_view(["POST"])
def proxy_generate_preparation(request):
    return runpod_post("/generate-preparation", dict(request.data), timeout=900)


@api_view(["POST"])
def proxy_generate_minutes(request):
    return runpod_post("/generate-minutes", dict(request.data), timeout=900)


@api_view(["POST"])
def proxy_chat(request):
    return runpod_post("/chat", dict(request.data), timeout=300)


@api_view(["POST"])
def proxy_ocr(request):
    upload = request.FILES.get("file")
    if not upload:
        return Response({"error": "OCR 파일을 업로드하세요."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        response = requests.post(
            runpod_url("/ocr", "ocr"),
            files={"file": (upload.name, upload.read(), upload.content_type or "application/octet-stream")},
            timeout=600,
        )
        response.raise_for_status()
        return Response(response.json(), status=status.HTTP_200_OK)
    except requests.RequestException as e:
        return Response({"error": f"RunPod 연결 실패: {str(e)}"}, status=status.HTTP_502_BAD_GATEWAY)


@api_view(["POST"])
def preview_agendas(request):
    title = str(request.data.get("title") or "").strip()
    project_id = request.data.get("project_id")
    if not title:
        return Response({"error": "회의 주제를 입력하세요."}, status=status.HTTP_400_BAD_REQUEST)

    previous_summary = request.data.get("previous_summary")
    if previous_summary is None and project_id:
        previous_summary = previous_summary_text(project_id)
    if project_id:
        try:
            project_context = Project.objects.get(project_id=project_id).context_summary
        except Project.DoesNotExist:
            project_context = ""
        previous_summary = "\n\n".join(part for part in [previous_summary or "", project_context] if part)

    payload = {
        "title": title,
        "previous_summary": previous_summary or "",
        "ocr_text": request.data.get("ocr_text", ""),
    }
    response = runpod_post("/generate-agendas", payload, timeout=300)
    if response.status_code < 400:
        return response

    fallback = [
        {"title": "회의 목표 및 범위 확정", "content": f"{title}의 목표, 산출물, 완료 기준을 정리한다."},
        {"title": "담당자별 준비사항 확인", "content": "참여자의 직무 기준으로 필요한 사전 자료와 역할을 확인한다."},
        {"title": "후속 업무 및 일정 합의", "content": "회의 이후 생성될 Task, 담당자, 마감일 기준을 합의한다."},
    ]
    return Response({"result": {"agendas": fallback}, "fallback_reason": response.data}, status=status.HTTP_200_OK)


@api_view(["GET", "POST"])
def meeting_list(request):
    if request.method == "POST":
        project_id = request.data.get("project_id")
        participant_ids = request.data.get("participant_project_user_ids") or request.data.get("participants") or []
        if not isinstance(participant_ids, list):
            return Response({"error": "참여자 목록은 배열이어야 합니다."}, status=status.HTTP_400_BAD_REQUEST)
        try:
            participant_ids = [int(item) for item in participant_ids]
        except (TypeError, ValueError):
            return Response({"error": "참여자 ID 형식이 올바르지 않습니다."}, status=status.HTTP_400_BAD_REQUEST)

        title = str(request.data.get("title") or "").strip()
        location = str(request.data.get("location") or "").strip()
        meeting_at = parse_meeting_at(request.data.get("meeting_at") or request.data.get("meeting_datetime"))
        if not title or not project_id or not meeting_at or not participant_ids:
            return Response(
                {"error": "회의 주제, 프로젝트, 일시, 참여자를 모두 입력하세요."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            project = Project.objects.get(project_id=project_id)
        except Project.DoesNotExist:
            return Response({"error": "프로젝트를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

        project_users = list(
            ProjectUsers.objects.filter(project=project, project_users_id__in=participant_ids)
            .select_related("user")
            .order_by("project_users_id")
        )
        if len(project_users) != len(set(participant_ids)):
            return Response({"error": "프로젝트에 없는 참여자가 포함되어 있습니다."}, status=status.HTTP_400_BAD_REQUEST)

        creator = ProjectUsers.objects.filter(
            project=project,
            project_users_id=request.data.get("creator_project_user_id") or participant_ids[0],
        ).first()
        if creator is None:
            creator = project_users[0]

        meeting = Meeting.objects.create(
            project=project,
            meeting_users=creator,
            title=title,
            location=location,
            meeting_at=meeting_at,
            meeting_document="",
            meeting_summary="",
            is_meeting=False,
        )
        record = Record.objects.create(meeting=meeting)
        for project_user in project_users:
            MeetingUsers.objects.create(meeting=meeting, user=project_user.user, record=record)

        agendas = request.data.get("agendas") or []
        if isinstance(agendas, list):
            for agenda in agendas:
                text = str(agenda.get("title") if isinstance(agenda, dict) else agenda).strip()
                content = str(agenda.get("content") or "").strip() if isinstance(agenda, dict) else ""
                value = f"{text}: {content}" if text and content else text or content
                if value:
                    MeetingAgendas.objects.create(meeting=meeting, content=value)

        return Response(serialize_meeting_detail(meeting), status=status.HTTP_201_CREATED)

    meetings = Meeting.objects.all().order_by("-meeting_at")
    serializer = MeetingSerializer(meetings, many=True)
    return Response(serializer.data)


@api_view(["GET"])
def meeting_detail(request, meeting_id):
    try:
        meeting = Meeting.objects.get(meeting_id=meeting_id)
    except Meeting.DoesNotExist:
        return Response({"error": "회의를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)
    return Response(serialize_meeting_detail(meeting))


@api_view(["POST"])
def generate_agendas(request, meeting_id):
    try:
        meeting = Meeting.objects.get(meeting_id=meeting_id)
    except Meeting.DoesNotExist:
        return Response({"error": "회의를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

    try:
        response = requests.post(
            runpod_url("/generate-agendas"),
            json={
                "title": meeting.title,
                "previous_summary": meeting.project.context_summary or previous_summary_text(meeting.project_id, meeting.meeting_id),
                "ocr_text": request.data.get("ocr_text", "") if hasattr(request, "data") else "",
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return Response({"error": f"RunPod 연결 실패: {str(e)}"}, status=status.HTTP_502_BAD_GATEWAY)

    result = data.get("result", data) if isinstance(data, dict) else {}
    agendas = result.get("agendas", []) if isinstance(result, dict) else []
    if not isinstance(agendas, list):
        agendas = []

    MeetingAgendas.objects.filter(meeting=meeting).delete()
    saved = []
    for item in agendas:
        if isinstance(item, dict):
            title = str(item.get("title") or "").strip()
            content = str(item.get("content") or "").strip()
            text = f"{title}: {content}" if title and content else title or content
        else:
            text = str(item).strip()
        if text:
            MeetingAgendas.objects.create(meeting=meeting, content=text)
            saved.append(text)

    return Response({"meeting_id": meeting_id, "agendas": saved, "raw": result}, status=status.HTTP_200_OK)


@api_view(["POST"])
def generate_preparation(request, meeting_id):
    try:
        meeting = Meeting.objects.get(meeting_id=meeting_id)
    except Meeting.DoesNotExist:
        return Response({"error": "회의를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

    participants = [
        {"name": item.user.name, "work": item.user.work}
        for item in MeetingUsers.objects.filter(meeting=meeting).select_related("user").order_by("meeting_users_id")
    ]
    agendas = list(MeetingAgendas.objects.filter(meeting=meeting).values_list("content", flat=True))
    previous_meetings = previous_meeting_sources(meeting)
    internal_documents = internal_document_sources(meeting)
    payload = {
        "title": meeting.title,
        "project_context": meeting.project.context_summary or "",
        "participants": participants,
        "agendas": agendas,
    }
    try:
        response = requests.post(
            runpod_url("/generate-preparation"),
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        agenda_text = "\n".join(f"- {agenda}" for agenda in agendas) or "- 등록된 기초안건이 없습니다."
        participant_text = "\n".join(f"- {item['name']} / {item['work']}" for item in participants)
        source_text = "\n".join(
            f"- {item['source']}: {item['text'][:160]}"
            for item in previous_meetings + internal_documents
        )
        document = (
            f"# {meeting.title} 준비자료\n\n"
            f"## 회의 정보\n- 일시: {meeting.meeting_at:%Y-%m-%d %H:%M}\n- 장소: {meeting.location}\n\n"
            f"## 참여자 직무\n{participant_text}\n\n"
            f"## 기초안건\n{agenda_text}\n\n"
            f"## 참고자료\n{source_text or '- 참고자료가 없습니다.'}"
        )
        MeetingPreparation.objects.update_or_create(meeting=meeting, defaults={"document": document})
        return Response(
            {
                "meeting_id": meeting_id,
                "document": document,
                "sources": {
                    "previous_meetings": previous_meetings,
                    "internal_documents": internal_documents,
                },
                "fallback_reason": f"RunPod 연결 실패: {str(e)}",
            },
            status=status.HTTP_200_OK,
        )

    result = data.get("result", data) if isinstance(data, dict) else {}
    document = ""
    if isinstance(result, dict):
        document = str(result.get("text") or result.get("document") or "")
        sections = result.get("sections")
        if not document and isinstance(sections, list):
            document = "\n\n".join(
                f"{section.get('title', '')}\n{section.get('content', '')}".strip()
                for section in sections
                if isinstance(section, dict)
            )
    MeetingPreparation.objects.update_or_create(meeting=meeting, defaults={"document": document})
    return Response(
        {
            "meeting_id": meeting_id,
            "document": document,
            "raw": result,
            "sources": result.get("sources", []) if isinstance(result, dict) else [],
        },
        status=status.HTTP_200_OK,
    )


@api_view(["POST"])
def meeting_chat(request, meeting_id):
    try:
        meeting = Meeting.objects.get(meeting_id=meeting_id)
    except Meeting.DoesNotExist:
        return Response({"error": "회의를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

    question = str(request.data.get("question") or "").strip()
    if not question:
        return Response({"error": "질문을 입력하세요."}, status=status.HTTP_400_BAD_REQUEST)

    try:
        response = requests.post(
            runpod_url("/chat"),
            json={
                "question": question,
                "history": request.data.get("history", []),
            },
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return Response(
            {
                "meeting_id": meeting_id,
                "result": local_meeting_chat_answer(meeting, question),
                "runpod_error": f"RunPod 연결 실패: {str(e)}",
            },
            status=status.HTTP_200_OK,
        )

    result = data.get("result", data) if isinstance(data, dict) else data
    return Response({"meeting_id": meeting_id, "result": result}, status=status.HTTP_200_OK)


# ── 회의 시작 ──────────────────────────────────────────────────────────────
# POST /api/meetings/<meeting_id>/start/
# 프론트가 이 요청을 받으면 동시에 마이크 녹음을 시작함
@api_view(["POST"])
def start_meeting(request, meeting_id):
    try:
        meeting = Meeting.objects.get(meeting_id=meeting_id)
    except Meeting.DoesNotExist:
        return Response({"error": "회의를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

    # 이미 진행 중인 회의라면 중복 시작 방지
    if meeting.is_meeting:
        return Response({"error": "이미 진행 중인 회의입니다."}, status=status.HTTP_400_BAD_REQUEST)

    # 회의 상태를 진행 중으로 변경
    meeting.is_meeting = True
    meeting.save()

    # Record 테이블에 row 생성 (녹음 파일은 아직 없으므로 비워둠)
    record = Record.objects.create(meeting=meeting)

    return Response({
        "message": "회의가 시작되었습니다.",
        "meeting_id": meeting_id,
        "record_id": record.record_id,
    }, status=status.HTTP_200_OK)


# ── 회의 종료 ──────────────────────────────────────────────────────────────
# POST /api/meetings/<meeting_id>/end/
# 프론트에서 녹음 파일(audio)을 multipart/form-data로 함께 전송
@api_view(["POST"])
def end_meeting(request, meeting_id):
    try:
        meeting = Meeting.objects.get(meeting_id=meeting_id)
    except Meeting.DoesNotExist:
        return Response({"error": "회의를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

    # 회의 상태를 종료로 변경
    meeting.is_meeting = False
    meeting.save()

    # 프론트에서 녹음 파일을 보냈다면 저장
    audio_file = request.FILES.get("audio") or request.FILES.get("audio_file")
    record = Record.objects.filter(meeting=meeting).last()
    if audio_file:
        # media/records/ 폴더에 meeting_id로 구분해서 저장
        save_dir = os.path.join(settings.MEDIA_ROOT, "records", str(meeting_id))
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, audio_file.name)

        with open(file_path, "wb+") as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # Record 테이블에 파일 경로 저장
        if record:
            record.record_path = file_path
            record.save()

    transcript = str(request.data.get("record_row_text") or request.data.get("transcript") or "").strip()
    if not transcript and record and record.record_path:
        try:
            transcript = runpod_transcribe_audio(record.record_path)
        except requests.RequestException as exc:
            return Response({"error": f"STT 변환 실패: {str(exc)}"}, status=status.HTTP_502_BAD_GATEWAY)

    if transcript:
        if record is None:
            record = Record.objects.create(meeting=meeting)
        record.record_row_text = transcript
        record.save(update_fields=["record_row_text"])

    return Response({
        "message": "회의가 종료되었습니다.",
        "meeting_id": meeting_id,
        "record_id": record.record_id if record else None,
        "has_transcript": bool(transcript),
        "record_row_text": transcript,
    }, status=status.HTTP_200_OK)


# ── 회의록 생성 ────────────────────────────────────────────────────────────
# POST /api/meetings/<meeting_id>/minutes/
# Django가 RunPod에 녹음 파일을 보내고, 받은 회의록 + 태스크를 DB에 저장
@api_view(["POST"])
def generate_minutes(request, meeting_id):
    try:
        meeting = Meeting.objects.get(meeting_id=meeting_id)
    except Meeting.DoesNotExist:
        return Response({"error": "회의를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

    # 해당 회의의 STT 텍스트 조회 (다른 팀원이 STT 변환 후 record_row_text에 저장)
    record = Record.objects.filter(meeting=meeting).last()
    if not record or not record.record_row_text:
        return Response({"error": "변환된 텍스트가 없습니다. STT 처리가 완료됐는지 확인해주세요."}, status=status.HTTP_400_BAD_REQUEST)

    # RunPod에 텍스트 전송 → 회의록 + 태스크 생성 요청
    runpod_url = settings.RUNPOD_MINUTES_URL
    try:
        response = requests.post(
            runpod_url,
            json={
                "text": record.record_row_text,
                "meeting_id": str(meeting.meeting_id),
                "project_id": str(meeting.project_id),
                "title": meeting.title,
                "meeting_datetime": meeting.meeting_at.isoformat() if meeting.meeting_at else "",
                "location": meeting.location,
                "project_context": meeting.project.context_summary or "",
            },
            timeout=300,  # LLM 처리 시간 고려해서 넉넉하게
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return Response({"error": f"RunPod 연결 실패: {str(e)}"}, status=status.HTTP_502_BAD_GATEWAY)

    # RunPod 응답 파싱
    # 응답 형식: {"result": {"summary": "...", "cotent": "...", "todo_list": [...]}, ...}
    result = data.get("result", data) if isinstance(data, dict) else {}
    if not isinstance(result, dict):
        result = {}
    summary = result.get("summary", "")
    cotent = result.get("cotent") or result.get("content") or ""
    todo_list = result.get("todo_list", [])
    if not isinstance(todo_list, list):
        todo_list = []

    # ① 회의록 텍스트 저장
    meeting.meeting_document = cotent
    meeting.meeting_summary = summary
    meeting.save(update_fields=["meeting_document", "meeting_summary"])
    append_project_context(meeting.project, meeting, summary)

    # ② todo_list → MeetingTask 테이블에 저장
    created_tasks = []
    skipped_tasks = []
    MeetingTask.objects.filter(meeting=meeting).delete()

    for todo in todo_list:
        if not isinstance(todo, dict):
            skipped_tasks.append({"title": "", "reason": "Task 형식이 올바르지 않습니다."})
            continue
        owner_name = todo.get("owner", "")
        title = todo.get("title", "")
        content = todo.get("content", "")
        due_date_str = todo.get("due_date", "")
        priority_str = todo.get("priority", "Medium")

        # 담당자 이름으로 MeetingUsers 조회
        meeting_user = MeetingUsers.objects.filter(
            meeting=meeting,
            user__name=owner_name
        ).first()

        if not meeting_user:
            # 담당자를 찾지 못하면 건너뜀
            skipped_tasks.append({"title": title, "reason": f"'{owner_name}' 담당자를 찾을 수 없음"})
            continue

        # due_date 문자열 → datetime 변환
        try:
            due_date = parse_meeting_at(due_date_str) or timezone.make_aware(
                datetime.strptime(due_date_str, "%Y-%m-%d"),
                timezone.get_current_timezone(),
            )
        except (ValueError, TypeError):
            due_date = timezone.now()

        # priority 문자열 → 정수 변환
        priority_int = PRIORITY_MAP.get(priority_str, 2)  # 기본값 Medium(2)

        MeetingTask.objects.create(
            meeting=meeting,
            meeting_users=meeting_user,
            title=title,
            content=content,
            due_date=due_date,
            priority=priority_int,
            status=0,  # 기본 상태: 미완료
        )
        created_tasks.append({
            "title": title,
            "content": content,
            "owner": owner_name,
            "due_date": due_date_str,
            "priority": priority_str,
        })

    return Response({
        "message": "회의록이 생성되었습니다.",
        "meeting_id": meeting_id,
        "summary": summary,
        "cotent": cotent,
        "content": cotent,
        "created_tasks": created_tasks,
        "skipped_tasks": skipped_tasks,
        "qdrant_ingest": result.get("qdrant_ingest"),
        "qdrant_ingest_error": result.get("qdrant_ingest_error"),
    }, status=status.HTTP_200_OK)
