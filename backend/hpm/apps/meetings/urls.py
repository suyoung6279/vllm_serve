from django.urls import path

from . import views


urlpatterns = [
    path("", views.meeting_list),
    path("agendas/preview/", views.preview_agendas),
    path("<int:meeting_id>/", views.meeting_detail),
    path("<int:meeting_id>/agendas/generate/", views.generate_agendas),
    path("<int:meeting_id>/preparation/generate/", views.generate_preparation),
    path("<int:meeting_id>/chat/", views.meeting_chat),
    path("<int:meeting_id>/start/", views.start_meeting),
    path("<int:meeting_id>/end/", views.end_meeting),
    path("<int:meeting_id>/minutes/", views.generate_minutes),
]
