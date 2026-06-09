from django.urls import path

from . import views


urlpatterns = [
    path("<int:project_id>/users/", views.project_users),
]
