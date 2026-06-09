import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  createMeeting,
  getMeetings,
  getProjectUsers,
  previewAgendas,
  type AgendaItem,
} from "../../features/meeting/api";
import type { Meeting, ProjectUser } from "../../types/meeting";

const PROJECT_ID = 1;

function agendaLabel(agenda: string | AgendaItem) {
  if (typeof agenda === "string") return agenda;
  return agenda.content ? `${agenda.title}: ${agenda.content}` : agenda.title;
}

export default function MeetingListPage() {
  const navigate = useNavigate();
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [members, setMembers] = useState<ProjectUser[]>([]);
  const [title, setTitle] = useState("회의록 자동화 기능 개발 회의");
  const [location, setLocation] = useState("3층 회의실 A");
  const [meetingAt, setMeetingAt] = useState("2026-06-07 15:00");
  const [selectedMemberIds, setSelectedMemberIds] = useState<number[]>([]);
  const [agendas, setAgendas] = useState<AgendaItem[]>([]);
  const [selectedAgendaIndexes, setSelectedAgendaIndexes] = useState<number[]>([]);
  const [loading, setLoading] = useState("");

  const selectedAgendas = useMemo(
    () => agendas.filter((_, index) => selectedAgendaIndexes.includes(index)),
    [agendas, selectedAgendaIndexes]
  );

  const load = async () => {
    const [meetingData, memberData] = await Promise.all([
      getMeetings(),
      getProjectUsers(PROJECT_ID),
    ]);
    setMeetings(meetingData);
    setMembers(memberData);
    if (selectedMemberIds.length === 0) {
      setSelectedMemberIds(memberData.slice(0, 4).map((item) => item.project_users_id));
    }
  };

  useEffect(() => {
    load().catch((error) => alert(error.message));
  }, []);

  const handlePreviewAgendas = async () => {
    setLoading("agendas");
    try {
      const items = await previewAgendas(PROJECT_ID, title);
      setAgendas(items);
      setSelectedAgendaIndexes(items.map((_, index) => index));
    } finally {
      setLoading("");
    }
  };

  const handleCreateMeeting = async () => {
    if (!selectedMemberIds.length) {
      alert("참여자를 선택하세요.");
      return;
    }
    setLoading("create");
    try {
      const meeting = await createMeeting({
        project_id: PROJECT_ID,
        title,
        location,
        meeting_at: meetingAt,
        participant_project_user_ids: selectedMemberIds,
        agendas: selectedAgendas,
      });
      navigate(`/meeting/${meeting.meeting_id}`);
    } finally {
      setLoading("");
    }
  };

  const toggleMember = (id: number) => {
    setSelectedMemberIds((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id]
    );
  };

  const toggleAgenda = (index: number) => {
    setSelectedAgendaIndexes((current) =>
      current.includes(index) ? current.filter((item) => item !== index) : [...current, index]
    );
  };

  return (
    <div className="mx-auto flex max-w-6xl gap-6 px-6 pt-24 pb-10">
      <section className="w-[440px] shrink-0">
        <h1 className="text-2xl font-bold text-gray-950">회의 생성</h1>
        <div className="mt-5 space-y-4">
          <label className="block">
            <span className="text-sm font-semibold text-gray-700">회의 주제</span>
            <input className="mt-2 w-full rounded-lg border bg-white px-3 py-2" value={title} onChange={(event) => setTitle(event.target.value)} />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="text-sm font-semibold text-gray-700">장소</span>
              <input className="mt-2 w-full rounded-lg border bg-white px-3 py-2" value={location} onChange={(event) => setLocation(event.target.value)} />
            </label>
            <label className="block">
              <span className="text-sm font-semibold text-gray-700">일시</span>
              <input className="mt-2 w-full rounded-lg border bg-white px-3 py-2" value={meetingAt} onChange={(event) => setMeetingAt(event.target.value)} />
            </label>
          </div>

          <div>
            <p className="text-sm font-semibold text-gray-700">참여자</p>
            <div className="mt-2 grid gap-2">
              {members.map((member) => (
                <label key={member.project_users_id} className="flex items-center justify-between rounded-lg border bg-white px-3 py-2">
                  <span>
                    <b>{member.name}</b>
                    <span className="ml-2 text-sm text-gray-500">{member.work}</span>
                  </span>
                  <input type="checkbox" checked={selectedMemberIds.includes(member.project_users_id)} onChange={() => toggleMember(member.project_users_id)} />
                </label>
              ))}
            </div>
          </div>

          <button className="w-full rounded-lg bg-gray-950 px-4 py-3 font-semibold text-white disabled:opacity-50" onClick={handlePreviewAgendas} disabled={loading !== ""}>
            {loading === "agendas" ? "기초안건 생성 중" : "기초안건 생성"}
          </button>

          {agendas.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm font-semibold text-gray-700">기초안건 선택</p>
              {agendas.map((agenda, index) => (
                <label key={`${agenda.title}-${index}`} className="flex gap-3 rounded-lg border bg-white px-3 py-2">
                  <input type="checkbox" checked={selectedAgendaIndexes.includes(index)} onChange={() => toggleAgenda(index)} />
                  <span className="text-sm">{agendaLabel(agenda)}</span>
                </label>
              ))}
            </div>
          )}

          <button className="w-full rounded-lg bg-blue-600 px-4 py-3 font-semibold text-white disabled:opacity-50" onClick={handleCreateMeeting} disabled={loading !== ""}>
            {loading === "create" ? "회의 생성 중" : "회의 생성"}
          </button>
        </div>
      </section>

      <section className="min-w-0 flex-1">
        <h2 className="text-2xl font-bold text-gray-950">회의 목록</h2>
        <div className="mt-5 grid gap-3">
          {meetings.map((meeting) => (
            <button
              key={meeting.meeting_id}
              onClick={() => navigate(`/meeting/${meeting.meeting_id}`)}
              className="rounded-lg border bg-white p-4 text-left transition hover:border-blue-400"
            >
              <div className="flex items-center justify-between gap-4">
                <h3 className="font-semibold text-gray-950">{meeting.title}</h3>
                <span className="text-sm text-gray-500">{meeting.is_meeting ? "진행 중" : "대기/종료"}</span>
              </div>
              <p className="mt-2 text-sm text-gray-600">{meeting.location} · {meeting.meeting_at?.slice(0, 16).replace("T", " ")}</p>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
