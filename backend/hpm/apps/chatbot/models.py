from django.db import models
from apps.meetings.models import Meeting, MeetingUsers


class Chatbot(models.Model):
    chatbot_id = models.AutoField(primary_key=True)
    meeting = models.ForeignKey(Meeting, on_delete=models.CASCADE, db_column="meeting_id")
    meeting_users = models.ForeignKey(
        MeetingUsers,
        on_delete=models.CASCADE,
        db_column="meeting_users_id"
    )

    class Meta:
        db_table = "chatbot"


class ChatHistory(models.Model):
    chat_history_id = models.AutoField(primary_key=True)
    chat = models.ForeignKey(Chatbot, on_delete=models.CASCADE, db_column="chat_id")
    type = models.IntegerField()
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "chat_history"
