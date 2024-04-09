from django.urls import path
from . import views

urlpatterns = [

    path('api/generate_and_save_tent_map/', views.generate_and_save_tent_map, name='generate_and_save_tent_map'),
    path('api/generate_and_save_logistic_map/', views.generate_and_save_logistic_map, name='generate_and_save_logistic_map'),
    path('api/generate_and_save_complex_squaring_map/', views.generate_and_save_complex_squaring_map, name='generate_and_save_complex_squaring_map'),


]