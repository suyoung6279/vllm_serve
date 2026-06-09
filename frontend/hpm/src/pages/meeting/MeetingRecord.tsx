import { useRef, useState } from "react";
import axios from "axios";

import stop from "../../assets/stop.png";

export default function MeetingRecordPage() {
  const [recording, setRecording] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mediaRecorder = new MediaRecorder(stream);

      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        await uploadAudio(audioBlob);

        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setRecording(true);
    } catch (err) {
      console.error("마이크 접근 실패:", err);
      alert("마이크 권한을 허용해주세요.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop();
      setRecording(false);
    }
  };

  const uploadAudio = async (blob: Blob) => {
    const formData = new FormData();
    formData.append("audio_file", blob, "recording.webm");

    try {
      const response = await axios.post(
        "http://localhost:8000/api/voice/",
        formData
      );

      console.log("업로드 성공:", response.data);
    } catch (error) {
      console.error("업로드 실패:", error);
    }
  };

  return (
    <div className="grid h-screen grid-cols-2">
      <section className="flex flex-col items-center justify-center border-r border-gray-200 px-6">
        <p className="cafe24-font mb-10 text-3xl text-[#0A0A0A] text-center">
          {recording ? "회의가 녹음 중 입니다." : "녹음을 시작하세요."}
        </p>

        <button onClick={recording ? stopRecording : startRecording}>
          <p className="cafe24-font mb-10 text-6xl text-[#0A0A0A] text-center">
            0:00
          </p>
        </button>

        <img
          src={stop}
          alt="stop"
          onClick={recording ? stopRecording : startRecording}
          className="h-40 w-40 cursor-pointer"
        />
      </section>

      <section className="flex flex-col items-center justify-center px-6">
        <p className="cafe24-font mb-10 text-3xl text-gray-800 text-center">
          버튼을 눌러 회의를 시작하세요!
        </p>
      </section>
    </div>
  );
}
