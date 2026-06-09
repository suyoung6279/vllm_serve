import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import {
  endMeeting,
  generateMinutes,
  generatePreparation,
  getMeetingDetail,
  meetingChat,
  startMeeting,
} from "../../features/meeting/api";
import type { Meeting } from "../../types/meeting";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export default function MeetingRecordStartPage() {
  const { meetingId } = useParams();
  const navigate = useNavigate();
  const id = Number(meetingId);
  const [meeting, setMeeting] = useState<Meeting | null>(null);
  const [recording, setRecording] = useState(false);
  const [busy, setBusy] = useState("");
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [liveTranscript, setLiveTranscript] = useState("");
  const [elapsedSec, setElapsedSec] = useState(0);
  const [audioUrl, setAudioUrl] = useState("");
  const [recordingError, setRecordingError] = useState("");
  const [sttReady, setSttReady] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);

  const formatElapsed = (seconds: number) => {
    const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
    const rest = Math.floor(seconds % 60).toString().padStart(2, "0");
    return `${minutes}:${rest}`;
  };

  const preferredMimeType = () => {
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/mp4",
      "audio/mpeg",
    ];
    return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || "";
  };

  const loadMeeting = async () => {
    const data = await getMeetingDetail(id);
    setMeeting(data);
  };

  useEffect(() => {
    if (!Number.isFinite(id)) return;
    loadMeeting().catch((error) => alert(error.message));
  }, [id]);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const handleGeneratePreparation = async () => {
    setBusy("preparation");
    try {
      await generatePreparation(id);
      await loadMeeting();
    } finally {
      setBusy("");
    }
  };

  const startRecording = async () => {
    audioChunksRef.current = [];
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl("");
    }
    setElapsedSec(0);
    setRecordingError("");
    setSttReady(false);
    setLiveTranscript("");
    setBusy("start");
    try {
      await startMeeting(id);
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = preferredMimeType();
      const mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      mediaRecorder.onstop = async () => {
        const audioBlob = audioChunksRef.current.length
          ? new Blob(audioChunksRef.current, { type: "audio/webm" })
          : null;
        if (audioBlob) {
          setAudioUrl(URL.createObjectURL(audioBlob));
        }
        if (timerRef.current) {
          window.clearInterval(timerRef.current);
          timerRef.current = null;
        }
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        await finishMeeting(audioBlob);
      };
      mediaRecorder.onerror = () => {
        setRecordingError("녹음 중 오류가 발생했습니다.");
      };
      mediaRecorder.start(1000);
      timerRef.current = window.setInterval(() => {
        setElapsedSec((value) => value + 1);
      }, 1000);
      setRecording(true);
      await loadMeeting();
    } catch (error) {
      setRecordingError(error instanceof Error ? error.message : "녹음을 시작하지 못했습니다.");
      alert(error instanceof Error ? error.message : "녹음을 시작하지 못했습니다.");
    } finally {
      setBusy("");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      setBusy("finish");
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const finishMeeting = async (audioBlob: Blob | null) => {
    try {
      const result = await endMeeting(id, audioBlob, liveTranscript);
      const refreshed = await getMeetingDetail(id);
      setMeeting(refreshed);
      const transcript = result.record_row_text || refreshed.record?.record_row_text || "";
      setLiveTranscript(transcript);
      setSttReady(Boolean(transcript));
      if (!transcript.trim()) {
        setRecordingError("STT는 완료됐지만 원문이 비어 있습니다. 마이크 입력이 없었거나 STT 서버가 빈 결과를 반환했습니다.");
      }
    } catch (error) {
      setRecordingError(error instanceof Error ? error.message : "회의 종료 처리에 실패했습니다.");
      alert(error instanceof Error ? error.message : "회의 종료 처리에 실패했습니다.");
    } finally {
      setBusy("");
    }
  };

  const handleGenerateMinutes = async () => {
    if (!liveTranscript.trim()) {
      alert("회의 원문이 있어야 회의록을 작성할 수 있습니다.");
      return;
    }
    setBusy("minutes");
    try {
      await generateMinutes(id);
      navigate(`/meeting/${id}/result`);
    } catch (error) {
      alert(error instanceof Error ? error.message : "회의록 작성에 실패했습니다.");
    } finally {
      setBusy("");
    }
  };

  const sendQuestion = async () => {
    const value = question.trim();
    if (!value) return;
    const nextHistory: ChatMessage[] = [...chatHistory, { role: "user", content: value }];
    setChatHistory(nextHistory);
    setQuestion("");
    try {
      const result = await meetingChat(id, value, nextHistory);
      const answer = result?.answer || JSON.stringify(result);
      setChatHistory([...nextHistory, { role: "assistant", content: answer }]);
    } catch {
      setChatHistory([...nextHistory, { role: "assistant", content: "답변 생성에 실패했습니다." }]);
    }
  };

  if (!meeting) {
    return <div className="px-6 pt-24">회의 정보를 불러오는 중입니다.</div>;
  }

  return (
    <div className="mx-auto grid max-w-6xl grid-cols-[1fr_400px] gap-6 px-6 pt-24 pb-10">
      <section className="min-w-0">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-gray-950">{meeting.title}</h1>
            <p className="mt-2 text-sm text-gray-600">
              {meeting.location} · {meeting.meeting_at?.slice(0, 16).replace("T", " ")}
            </p>
          </div>
          <button
            onClick={handleGeneratePreparation}
            disabled={busy !== ""}
            className="rounded-lg bg-gray-950 px-4 py-2 font-semibold text-white disabled:opacity-50"
          >
            {busy === "preparation" ? "생성 중" : "준비자료 생성"}
          </button>
        </div>

        <div className="mt-6 grid gap-4">
          <section>
            <h2 className="font-semibold text-gray-900">참여자</h2>
            <div className="mt-2 flex flex-wrap gap-2">
              {meeting.participants?.map((member) => (
                <span key={member.meeting_users_id ?? member.project_users_id ?? member.user_id} className="rounded-lg border bg-white px-3 py-2 text-sm">
                  {member.name} · {member.work}
                </span>
              ))}
            </div>
          </section>

          <section>
            <h2 className="font-semibold text-gray-900">기초안건</h2>
            <ul className="mt-2 grid gap-2">
              {meeting.agendas?.map((agenda, index) => (
                <li key={`${agenda}-${index}`} className="rounded-lg border bg-white px-3 py-2 text-sm">{agenda}</li>
              ))}
            </ul>
          </section>

          <section>
            <h2 className="font-semibold text-gray-900">준비자료</h2>
            <pre className="mt-2 max-h-[360px] overflow-auto whitespace-pre-wrap rounded-lg border bg-white p-4 text-sm leading-6">
              {meeting.preparation?.document || "아직 준비자료가 없습니다. 준비자료 생성을 먼저 실행하세요."}
            </pre>
          </section>

          <section>
            <div className="flex items-center justify-between">
              <h2 className="font-semibold text-gray-900">회의 텍스트</h2>
            </div>
            <div className="mt-2 rounded-lg border bg-white p-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-3xl font-bold tabular-nums text-gray-950">{formatElapsed(elapsedSec)}</p>
                  <p className="mt-1 text-sm text-gray-500">
                    {recording ? "마이크 입력을 녹음 중입니다." : busy === "finish" ? "녹음 파일을 업로드하고 STT 원문을 추출 중입니다." : sttReady ? "STT 원문이 준비되었습니다. 아래 내용을 확인한 뒤 회의록을 작성하세요." : "녹음 시작을 누르면 실제 음성이 저장됩니다."}
                  </p>
                </div>
                <button
                  onClick={recording ? stopRecording : startRecording}
                  disabled={busy !== ""}
                  className={`h-14 rounded-lg px-6 font-semibold text-white disabled:opacity-50 ${recording ? "bg-red-600" : "bg-blue-600"}`}
                >
                  {recording ? "녹음 종료" : busy === "finish" ? "처리 중" : "녹음 시작"}
                </button>
              </div>
              {recording && (
                <div className="mt-4 h-2 overflow-hidden rounded-full bg-gray-100">
                  <div className="h-full w-1/2 animate-pulse rounded-full bg-red-500" />
                </div>
              )}
              {audioUrl && (
                <audio controls src={audioUrl} className="mt-4 w-full" />
              )}
              {recordingError && (
                <p className="mt-3 text-sm text-red-600">{recordingError}</p>
              )}
            </div>
            <textarea
              value={liveTranscript}
              onChange={(event) => setLiveTranscript(event.target.value)}
              placeholder="녹음 종료 시 실제 음성 파일을 RunPod /stt로 보내 원문을 추출합니다. STT를 우회해야 할 때만 직접 텍스트를 입력하세요."
              className="mt-2 h-44 w-full rounded-lg border bg-white p-3 text-sm leading-6"
            />
            <div className="mt-3 flex items-center justify-end gap-3">
              <button
                onClick={handleGenerateMinutes}
                disabled={busy !== "" || !liveTranscript.trim()}
                className="rounded-lg bg-gray-950 px-5 py-2 font-semibold text-white disabled:opacity-50"
              >
                {busy === "minutes" ? "회의록 작성 중" : "이 원문으로 회의록 작성"}
              </button>
            </div>
          </section>
        </div>
      </section>

      <aside className="flex h-[calc(100vh-120px)] flex-col rounded-lg border bg-white p-4">
        <h2 className="text-lg font-bold text-gray-950">회의 도우미 챗봇</h2>
        <div className="mt-4 flex-1 space-y-3 overflow-auto">
          {chatHistory.length === 0 && (
            <p className="text-sm text-gray-500">녹음 중 이전회의록, 내부자료, 외부자료 기반으로 질문할 수 있습니다.</p>
          )}
          {chatHistory.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={`rounded-lg px-3 py-2 text-sm ${message.role === "user" ? "ml-10 bg-blue-600 text-white" : "mr-10 bg-gray-100 text-gray-900"}`}
            >
              {message.content}
            </div>
          ))}
        </div>
        <div className="mt-4 flex gap-2">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") sendQuestion();
            }}
            disabled={!recording}
            placeholder={recording ? "질문 입력" : "녹음 시작 후 사용"}
            className="min-w-0 flex-1 rounded-lg border px-3 py-2 text-sm"
          />
          <button onClick={sendQuestion} disabled={!recording} className="rounded-lg bg-gray-950 px-4 py-2 text-sm font-semibold text-white disabled:opacity-50">
            전송
          </button>
        </div>
      </aside>
    </div>
  );
}
