from django.db import models


class Dept(models.Model):
    dept_id = models.AutoField(primary_key=True)
    dept_name = models.CharField(max_length=20)

    class Meta:
        db_table = "dept"


class Rank(models.Model):
    rank_id = models.AutoField(primary_key=True)
    rank_name = models.CharField(max_length=20)

    class Meta:
        db_table = "rank"


class Users(models.Model):
    users_id = models.AutoField(primary_key=True)
    dept = models.ForeignKey(Dept, on_delete=models.PROTECT, db_column="dept_id")
    rank = models.ForeignKey(Rank, on_delete=models.PROTECT, db_column="rank_id")

    emp_no = models.CharField(max_length=20)
    email = models.EmailField(max_length=255)
    name = models.CharField(max_length=90)
    work = models.CharField(max_length=150)

    password = models.CharField(max_length=255, default="abc123")
    account_status = models.IntegerField(default=0)
    status = models.IntegerField(default=0)
    account_id = models.CharField(max_length=255, null=True, blank=True, db_column="account_Id")
    role = models.CharField(max_length=20, default="USER")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "users"
