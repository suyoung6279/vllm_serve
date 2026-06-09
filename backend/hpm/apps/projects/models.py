from django.db import models
from apps.users.models import Users

class Project(models.Model):
    project_id = models.AutoField(primary_key=True)
    project_owner = models.ForeignKey(
        Users,
        on_delete=models.PROTECT,
        db_column="project_owner_id",
        related_name="owned_projects"
    )
    project_name = models.CharField(max_length=90)
    context_summary = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "project"


class ProjectUsers(models.Model):
    project_users_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(Users, on_delete=models.CASCADE, db_column="user_id")
    project = models.ForeignKey(Project, on_delete=models.CASCADE, db_column="project_id")

    class Meta:
        db_table = "project_users"
