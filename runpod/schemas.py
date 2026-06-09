from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MinutesRequest(BaseModel):
    text: str = Field(..., description="STT transcript text.")
    meeting_id: str = ""
    project_id: str = ""
    title: str = ""
    meeting_datetime: str = ""
    location: str = ""
    project_context: str = ""


class AgendaRequest(BaseModel):
    title: str
    previous_summary: str = ""
    ocr_text: str = ""


class PreparationRequest(BaseModel):
    title: str = Field(..., description="Meeting topic or title used to generate preparation material.")
    project_context: str = ""
    participants: list[dict[str, str]] = Field(default_factory=list)
    agendas: list[str] = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str
    context: str = ""
    history: list[dict[str, str]] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


class TextResponse(BaseModel):
    result: Any
    elapsed_sec: float
    model: str
