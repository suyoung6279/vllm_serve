from django.db import models
from apps.users.models import Users

class Meeting(models.Model):
    meeting_id = models.AutoField(primary_key=True)
    project = models.ForeignKey(
        "projects.Project",
        on_delete=models.CASCADE,
        db_column="project_id"
    )
    meeting_users = models.ForeignKey(
        "projects.ProjectUsers",
        on_delete=models.PROTECT,
        db_column="meeting_users_id"
    )

    title = models.CharField(max_length=90)
    location = models.CharField(max_length=150, blank=True)
    meeting_at = models.DateTimeField()
    meeting_document = models.TextField(null=True, blank=True)
    meeting_summary = models.TextField(null=True, blank=True)
    is_meeting = models.BooleanField(default=False)

    class Meta:
        db_table = "meeting"


class Record(models.Model):
    record_id = models.AutoField(primary_key=True)
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        db_column="meeting_id"
    )
    record_path = models.TextField(null=True, blank=True)
    record_row_text = models.TextField(null=True, blank=True)

    class Meta:
        db_table = "record"


class MeetingUsers(models.Model):
    meeting_users_id = models.AutoField(primary_key=True)
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        db_column="meeting_id"
    )
    user = models.ForeignKey(
        Users,
        on_delete=models.CASCADE,
        db_column="user_id"
    )
    record = models.ForeignKey(
        Record,
        on_delete=models.CASCADE,
        db_column="record_id"
    )

    class Meta:
        db_table = "meeting_users"


class MeetingAgendas(models.Model):
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        db_column="meeting_id"
    )
    content = models.TextField()

    class Meta:
        db_table = "meeting_agendas"


class MeetingPreparation(models.Model):
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        db_column="meeting_id"
    )
    document = models.TextField()

    class Meta:
        db_table = "meeting_preparation"


class OuterDocument(models.Model):
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        db_column="meeting_id"
    )
    path = models.TextField()

    class Meta:
        db_table = "outer_document"


class MeetingTask(models.Model):
    meeting_task_id = models.AutoField(primary_key=True)
    meeting = models.ForeignKey(
        Meeting,
        on_delete=models.CASCADE,
        db_column="meeting_id"
    )
    meeting_users = models.ForeignKey(
        MeetingUsers,
        on_delete=models.CASCADE,
        db_column="meeting_users_id"
    )

    title = models.CharField(max_length=255)
    content = models.TextField()
    due_date = models.DateTimeField()
    priority = models.IntegerField(null=True, blank=True)
    status = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "meeting_task"