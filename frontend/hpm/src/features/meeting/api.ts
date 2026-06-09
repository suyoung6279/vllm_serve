import axios from "axios";
import type { Meeting, ProjectUser } from "../../types/meeting";

const api = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
});

export interface AgendaItem {
  title: string;
  content?: string;
}

export interface CreateMeetingPayload {
  project_id: number;
  title: string;
  location: string;
  meeting_at: string;
  participant_project_user_ids: number[];
  agendas: Array<string | AgendaItem>;
}

export const getProjectUsers = async (projectId: number): Promise<ProjectUser[]> => {
  const response = await api.get<ProjectUser[]>(`/projects/${projectId}/users/`);
  return response.data;
};

export const previewAgendas = async (
  projectId: number,
  title: string
): Promise<AgendaItem[]> => {
  const response = await api.post("/meetings/agendas/preview/", {
    project_id: projectId,
    title,
  });
  return response.data.result?.agendas || response.data.agendas || [];
};

export const createMeeting = async (
  payload: CreateMeetingPayload
): Promise<Meeting> => {
  const response = await api.post<Meeting>("/meetings/", payload);
  return response.data;
};

export const getMeetings = async (): Promise<Meeting[]> => {
  const response = await api.get<Meeting[]>("/meetings/");
  return response.data;
};

export const getMeetingDetail = async (meetingId: number): Promise<Meeting> => {
  const response = await api.get<Meeting>(`/meetings/${meetingId}/`);
  return response.data;
};

export const generatePreparation = async (meetingId: number) => {
  const response = await api.post(`/meetings/${meetingId}/preparation/generate/`, {});
  return response.data;
};

export const startMeeting = async (meetingId: number) => {
  const response = await api.post(`/meetings/${meetingId}/start/`, {});
  return response.data;
};

export const endMeeting = async (
  meetingId: number,
  audio: Blob | null,
  transcript: string
): Promise<{
  message: string;
  meeting_id: number;
  record_id: number | null;
  has_transcript: boolean;
  record_row_text: string;
}> => {
  const formData = new FormData();
  if (audio) {
    formData.append("audio_file", audio, "recording.webm");
  }
  formData.append("transcript", transcript);
  const response = await api.post(`/meetings/${meetingId}/end/`, formData);
  return response.data;
};

export const generateMinutes = async (meetingId: number) => {
  const response = await api.post(`/meetings/${meetingId}/minutes/`, {});
  return response.data;
};

export const meetingChat = async (
  meetingId: number,
  question: string,
  history: Array<{ role: string; content: string }>
) => {
  const response = await api.post(`/meetings/${meetingId}/chat/`, {
    question,
    history,
  });
  return response.data.result;
};
