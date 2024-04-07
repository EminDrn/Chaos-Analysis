from django.urls import path
from . import views

urlpatterns = [

    path('api/generate_and_save_tent_map/', views.generate_and_save_tent_map, name='generate_and_save_tent_map'),
]