import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { getMeetingDetail } from "../../features/meeting/api";
import type { Meeting } from "../../types/meeting";

export default function MeetingRecordResultPage() {
  const { meetingId } = useParams();
  const [meeting, setMeeting] = useState<Meeting | null>(null);

  useEffect(() => {
    const id = Number(meetingId);
    if (!Number.isFinite(id)) return;
    getMeetingDetail(id).then(setMeeting).catch((error) => alert(error.message));
  }, [meetingId]);

  if (!meeting) {
    return <div className="px-6 pt-24">회의 결과를 불러오는 중입니다.</div>;
  }

  return (
    <div className="mx-auto max-w-6xl px-6 pt-24 pb-10">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-950">회의 결과</h1>
          <p className="mt-2 text-gray-700">{meeting.title}</p>
        </div>
        <Link to="/meeting" className="rounded-lg bg-gray-950 px-4 py-2 font-semibold text-white">
          목록으로
        </Link>
      </div>

      <div className="mt-6 grid grid-cols-[1fr_360px] gap-6">
        <section>
          <h2 className="font-semibold text-gray-900">회의록</h2>
          <pre className="mt-2 min-h-[420px] whitespace-pre-wrap rounded-lg border bg-white p-5 text-sm leading-6">
            {meeting.meeting_document || "회의록 본문이 아직 생성되지 않았습니다."}
          </pre>
        </section>

        <aside className="space-y-5">
          <section>
            <h2 className="font-semibold text-gray-900">핵심 요약</h2>
            <p className="mt-2 rounded-lg border bg-white p-4 text-sm leading-6">
              {meeting.meeting_summary || "요약이 없습니다."}
            </p>
          </section>

          <section>
            <h2 className="font-semibold text-gray-900">Task</h2>
            <div className="mt-2 grid gap-3">
              {meeting.tasks?.length ? (
                meeting.tasks.map((task) => (
                  <article key={task.meeting_task_id} className="rounded-lg border bg-white p-4">
                    <div className="flex items-start justify-between gap-3">
                      <h3 className="font-semibold text-gray-950">{task.title}</h3>
                      <span className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-600">{task.priority}</span>
                    </div>
                    <p className="mt-2 text-sm text-gray-700">{task.content}</p>
                    <p className="mt-3 text-xs text-gray-500">
                      담당자: {task.owner} · 마감: {task.due_date?.slice(0, 10)}
                    </p>
                  </article>
                ))
              ) : (
                <p className="rounded-lg border bg-white p-4 text-sm text-gray-500">생성된 Task가 없습니다.</p>
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
