from django.contrib import admin
from django.urls import path
from .views import index_view, upload_view
urlpatterns = [
    path("", index_view, name="index"),
    path("predict/", upload_view, name="upload")
]
