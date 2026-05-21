from django.urls import path
from . import views

app_name = 'project2'

urlpatterns = [
    path('', views.index, name='index'),
    path("update-model/", views.update_model, name="update_model"),
    path("get-samples/", views.get_samples, name="get_samples"),
    path("counterfactuals/", views.counterfactuals, name="counterfactuals"),
    path("feature-effects/", views.feature_effects, name="feature_effects"),
    path("get-features/", views.get_numerical_features, name="get_features"),
]
