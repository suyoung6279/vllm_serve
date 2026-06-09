from django.db import models
from apps.projects.models import Project, ProjectUsers

class Document(models.Model):
    document_id = models.AutoField(primary_key=True)
    project = models.ForeignKey(Project, on_delete=models.CASCADE, db_column="project_id")
    uploader = models.ForeignKey(
        ProjectUsers,
        on_delete=models.PROTECT,
        db_column="uploader_id"
    )

    title = models.CharField(max_length=255)
    path = models.CharField(max_length=500)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "document"