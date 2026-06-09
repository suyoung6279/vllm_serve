export interface ProjectUser {
  project_users_id: number;
  meeting_users_id?: number;
  user_id: number;
  name: string;
  email: string;
  dept: string;
  rank: string;
  work: string;
}

export interface MeetingTask {
  meeting_task_id: number;
  title: string;
  content: string;
  owner: string;
  due_date: string;
  priority: string;
  status: number | null;
}

export interface Meeting {
  meeting_id: number;
  title: string;
  location: string;
  meeting_at: string;
  meeting_document: string | null;
  meeting_summary: string | null;
  is_meeting: boolean;
  project: number;
  meeting_users: number;
  participants?: ProjectUser[];
  agendas?: string[];
  preparation?: {
    id: number;
    document: string;
  } | null;
  record?: {
    record_id: number;
    record_path: string | null;
    record_row_text: string | null;
  } | null;
  tasks?: MeetingTask[];
}
