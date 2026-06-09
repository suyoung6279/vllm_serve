from django.db import models
from apps.users.models import Users


class Notification(models.Model):
    notification_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(Users, on_delete=models.CASCADE, db_column="user_id")
    content = models.TextField()
    is_read = models.BooleanField(default=False)

    class Meta:
        db_table = "notification"