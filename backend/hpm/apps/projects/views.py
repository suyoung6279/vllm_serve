from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Project, ProjectUsers


@api_view(["GET"])
def project_users(request, project_id):
    try:
        Project.objects.get(project_id=project_id)
    except Project.DoesNotExist:
        return Response({"error": "프로젝트를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

    members = (
        ProjectUsers.objects.filter(project_id=project_id)
        .select_related("user", "user__dept", "user__rank")
        .order_by("project_users_id")
    )
    return Response(
        [
            {
                "project_users_id": item.project_users_id,
                "user_id": item.user.users_id,
                "name": item.user.name,
                "email": item.user.email,
                "dept": item.user.dept.dept_name,
                "rank": item.user.rank.rank_name,
                "work": item.user.work,
            }
            for item in members
        ]
    )
