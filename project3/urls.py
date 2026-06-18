from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('classifier-stats/', views.classifier_stats, name='classifier_stats'),
    path('expert-stats/', views.expert_stats, name='expert_stats'),
]