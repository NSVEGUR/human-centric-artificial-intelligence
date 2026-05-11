from django.urls import path
from . import views

app_name = 'project2'

urlpatterns = [
    path('', views.index, name='index'),
    path("update-model/", views.update_model, name="update_model"),
]
