from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('classifier-stats/', views.classifier_stats, name='classifier_stats'),
    path('expert-stats/', views.expert_stats, name='expert_stats'),
    path('deferral-stats/', views.deferral_stats_api, name='deferral_stats'),
    path('al-stats/', views.al_stats_api, name='al_stats'),
    # Task 5: interactive human labeling
    path('human-label/', views.human_label, name='human_label'),
    path('human-label/submit/', views.human_label_submit, name='human_label_submit'),
    path('human-label/strategy/', views.human_label_strategy, name='human_label_strategy'),
    path('human-label/reset/', views.human_label_reset, name='human_label_reset'),
    path('report.pdf', views.download_report, name='report'),
]
