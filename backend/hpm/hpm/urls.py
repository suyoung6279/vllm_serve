"""
URL configuration for hpm project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from apps.meetings.views import (
    app_config,
    index,
    proxy_chat,
    proxy_generate_agendas,
    proxy_generate_minutes,
    proxy_generate_preparation,
    proxy_ocr,
    runpod_health,
)

urlpatterns = [
    path('', index),
    path("api/config", app_config),
    path("api/health", runpod_health),
    path("api/ocr", proxy_ocr),
    path("api/generate-agendas", proxy_generate_agendas),
    path("api/generate-preparation", proxy_generate_preparation),
    path("api/generate-minutes", proxy_generate_minutes),
    path("api/chat", proxy_chat),
    path('admin/', admin.site.urls),
    path("api/meetings/", include("apps.meetings.urls")),
    path("api/projects/", include("apps.projects.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
