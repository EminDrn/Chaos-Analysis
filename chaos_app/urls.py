from django.urls import path
from . import views
urlpatterns = [

    path('api/generate_and_save_tent_map/', views.generate_and_save_tent_map, name="generate_and_save_tent_map"),
    path('api/generate_and_save_logistic_map/', views.generate_and_save_logistic_map, name='generate_and_save_logistic_map'),
    path('api/generate_and_save_complex_squaring_map/', views.generate_and_save_complex_squaring_map, name='generate_and_save_complex_squaring_map'),
    
    path('api/arnoldcat_map/', views.arnoldcat_map_api, name='arnoldcat_map_api'),
    path('api/bernoulli_map/', views.generate_and_save_bernoulli_map, name='generate_and_save_bernoulli_map'),
    path('api/lorenz96_map/', views.generate_and_save_L96_trajectory, name='generate_and_save_L96_trajectory'),

    path('api/lorenz_map/', views.lorenz_map, name='lorenz_map'),
    path('api/poincare_map/', views.poincare_map_view, name='poincare_map'),
    path('api/generate_tinkerbell_map/', views.generate_and_save_tinkerbell_map, name='generate_tinkerbell_map'),
    path('api/generate_logistic_map/', views.generate_and_save_logistic_map, name='generate_logistic_map'),

    path('api/generate_and_save_gauss_map/', views.generate_and_save_gauss_map,name='generate_and_save_gauss_map'),
    path('api/generate_and_save_lotka_volterra_map/', views.generate_and_save_lotka_volterra_map, name='generate_and_save_lotka_volterra_map'),

    path('api/generate_and_save_zaslavskii_map/', views.generate_and_save_zaslavskii_map, name='generate_and_save_zaslavskii_map'),
    path('api/generate_and_save_complex_squared_map/', views.generate_and_save_complex_squared_map, name='generate_and_save_complex_squared_map'),
    path('api/generate_and_save_genetic_algorithm_map/', views.generate_and_save_genetic_algorithm_map, name='generate_and_save_genetic_algorithm_map'),
    path('api/generate_and_save_ikeda_attractor/', views.generate_and_save_ikeda_attractor , name="generate_and_save_ikeda_attractor"),
    path('api/generate_and_save_gingerbread_man/', views.generate_and_save_gingerbread_man,
         name="generate_and_save_gingerbread_man"),

]

