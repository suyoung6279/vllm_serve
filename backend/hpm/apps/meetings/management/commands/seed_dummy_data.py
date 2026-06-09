from __future__ import annotations

from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.documents.models import Document
from apps.meetings.models import Meeting, MeetingAgendas, MeetingPreparation, MeetingUsers, Record
from apps.projects.models import Project, ProjectUsers
from apps.users.models import Dept, Rank, Users


class Command(BaseCommand):
    help = "Seed demo data for the meeting service flow."

    def handle(self, *args, **options):
        depts = {
            "DEV": Dept.objects.update_or_create(dept_name="개발팀", defaults={})[0],
            "PLAN": Dept.objects.update_or_create(dept_name="기획팀", defaults={})[0],
            "HR": Dept.objects.update_or_create(dept_name="인사팀", defaults={})[0],
        }
        ranks = {
            "LEAD": Rank.objects.update_or_create(rank_name="팀장", defaults={})[0],
            "SENIOR": Rank.objects.update_or_create(rank_name="대리", defaults={})[0],
            "STAFF": Rank.objects.update_or_create(rank_name="사원", defaults={})[0],
        }

        user_specs = [
            ("U001", "DEV", "LEAD", "HPM-001", "jiwon.kim@example.com", "김지원", "프로젝트 총괄", "jira-jiwon", "ADMIN"),
            ("U002", "DEV", "SENIOR", "HPM-002", "gyuho.kim@example.com", "김규호", "STT 및 백엔드 연동", "jira-gyuho", "USER"),
            ("U003", "PLAN", "SENIOR", "HPM-003", "jiwoo.ryu@example.com", "류지우", "프론트엔드 및 발표자료", "jira-jiwoo", "USER"),
            ("U004", "HR", "STAFF", "HPM-004", "sooyoung.park@example.com", "박수영", "검수 및 운영 정책", "jira-sooyoung", "USER"),
        ]

        users: dict[str, Users] = {}
        for key, dept, rank, emp_no, email, name, work, account_id, role in user_specs:
            user, _ = Users.objects.update_or_create(
                emp_no=emp_no,
                defaults={
                    "dept": depts[dept],
                    "rank": ranks[rank],
                    "email": email,
                    "name": name,
                    "work": work,
                    "password": "abc123",
                    "account_status": 1,
                    "status": 0,
                    "account_id": account_id,
                    "role": role,
                },
            )
            users[key] = user

        project, _ = Project.objects.update_or_create(
            project_name="AI 회의 관리 플랫폼 구축",
            defaults={"project_owner": users["U001"]},
        )

        project_users: dict[str, ProjectUsers] = {}
        for key, user_key in [("PU001", "U001"), ("PU002", "U002"), ("PU003", "U003"), ("PU004", "U004")]:
            project_users[key], _ = ProjectUsers.objects.get_or_create(project=project, user=users[user_key])

        previous_at = timezone.make_aware(timezone.datetime(2026, 6, 1, 14, 0, 0))
        previous, _ = Meeting.objects.update_or_create(
            title="회의록 자동화 1차 설계 회의",
            defaults={
                "project": project,
                "meeting_users": project_users["PU001"],
                "location": "온라인",
                "meeting_at": previous_at,
                "meeting_document": (
                    "STT 결과는 Record.record_row_text에 저장하고, 회의록 생성 모델은 요약과 todo_list를 함께 반환한다. "
                    "준비자료는 이전회의록, 내부자료, 외부뉴스를 함께 참고한다."
                ),
                "meeting_summary": "STT 저장 위치, 회의록 생성 결과 형식, 준비자료 생성에 사용할 자료 범위를 확정했다.",
                "is_meeting": False,
            },
        )
        previous_record, _ = Record.objects.update_or_create(
            meeting=previous,
            defaults={
                "record_path": "",
                "record_row_text": (
                    "김지원: STT 결과는 Record 테이블에 저장합시다. "
                    "김규호: 회의록 생성 모델에는 record_row_text를 넘기겠습니다. "
                    "류지우: 화면에서는 준비자료를 먼저 보여주고 녹음 버튼을 배치하겠습니다."
                ),
            },
        )
        for user in users.values():
            MeetingUsers.objects.get_or_create(meeting=previous, user=user, record=previous_record)

        meeting_at = timezone.make_aware(timezone.datetime(2026, 6, 7, 15, 0, 0))
        meeting, _ = Meeting.objects.update_or_create(
            title="회의록 자동화 기능 개발 회의",
            defaults={
                "project": project,
                "meeting_users": project_users["PU001"],
                "location": "3층 회의실 A",
                "meeting_at": meeting_at,
                "meeting_document": "",
                "meeting_summary": "",
                "is_meeting": False,
            },
        )
        record, _ = Record.objects.update_or_create(
            meeting=meeting,
            defaults={
                "record_path": "",
                "record_row_text": (
                    "김지원: 오늘은 회의록 자동화 기능의 개발 범위를 확정하겠습니다. "
                    "김규호: STT API 연동 테스트는 이번 주 금요일까지 진행하겠습니다. "
                    "류지우: 회의 결과 화면 초안은 다음 주 월요일까지 만들겠습니다. "
                    "박수영: 운영 정책과 검수 기준 문서를 정리하겠습니다."
                ),
            },
        )
        for user in users.values():
            MeetingUsers.objects.get_or_create(meeting=meeting, user=user, record=record)

        for agenda in ["STT API 연동 방식 결정", "회의 결과 화면 구성", "RunPod 생성 결과 저장 구조"]:
            MeetingAgendas.objects.get_or_create(meeting=meeting, content=agenda)

        MeetingPreparation.objects.update_or_create(
            meeting=meeting,
            defaults={
                "document": (
                    "# 회의록 자동화 기능 개발 회의 준비자료\n\n"
                    "이전 회의에서는 STT 결과 저장 위치와 회의록 생성 모델의 반환 형식을 확정했다. "
                    "이번 회의에서는 실제 화면 흐름, 녹음 종료 후 처리, 담당자 매핑 기준을 결정해야 한다."
                )
            },
        )

        for title, path in [
            ("회의 자동화 요구사항 명세서", "dummy/internal/meeting_requirements.md"),
            ("Task 담당자 매핑 정책", "dummy/internal/task_owner_policy.md"),
        ]:
            Document.objects.update_or_create(
                project=project,
                title=title,
                defaults={"uploader": project_users["PU001"], "path": path},
            )

        self.stdout.write(self.style.SUCCESS("Seeded demo meeting data."))
        self.stdout.write(f"Project id: {project.project_id}")
        self.stdout.write(f"Meeting id: {meeting.meeting_id}")
