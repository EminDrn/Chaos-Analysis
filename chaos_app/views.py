import json

import random
import math
import numpy as np
import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from django.core.files.base import ContentFile
from PIL.Image import open as load_pic, new as new_pic
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

#model import
from .models import plot_L96_trajectory,plot_bernoulli_map,arnoldcat_map


#numpy , django , matplotlib ,  scipy
@csrf_exempt
#*
def generate_and_save_tent_map(request):
    if request.method == 'POST':
        # İstek gövdesini JSON olarak ayrıştır
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        x0 = float(body['formData'].get('x0', 0.1))
        r = float(body['formData'].get('r', 1.65))
        iterations = int(body['formData'].get('iterations', 100))

        sequence = [x0]
        for _ in range(iterations - 1):
            x0 = tent_map(x0, r)
            sequence.append(x0)

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'tent_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.figure(figsize=(10, 6))
        plt.plot(sequence, 'b-', linewidth=0.5, marker='o', markersize=2)
        plt.title('Tent Map: r = {}'.format(r))
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = os.path.join('chaos_app', 'maps', 'tent_map.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)

def tent_map(x, r):
    if x < 0.5:
        return r * x
    else:
        return r * (1 - x)
#*
@csrf_exempt
def generate_and_save_ikeda_attractor(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        x0 = float(body['formData'].get('x0', 0.1))
        y0 = float(body['formData'].get('y0', 0.1))
        u = float(body['formData'].get('u', 0.9))
        iterations = int(body['formData'].get('iterations', 10000))

        x_values, y_values = plot_ikeda_attractor(x0, y0, u, iterations)

        # File path
        file_path = os.path.join('chaos_app', 'maps', 'ikeda_attractor.png')

        # Remove file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)

        # Plot and save the image
        plt.figure(figsize=(8, 6))
        plt.plot(x_values, y_values, '.', markersize=0.5)
        plt.title('Ikeda Chaotic Attractor')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(file_path)
        plt.close()

        # Return the saved file relative path
        plot_url = os.path.join('chaos_app', 'maps', 'ikeda_attractor.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)


def plot_ikeda_attractor(x0=0.1, y0=0.1, u=0.9, iterations=10000):
    def ikeda_map(x, y, u):
        t = 0.4 - 6 / (1 + x ** 2 + y ** 2)
        x_next = 1 + u * (x * np.cos(t) - y * np.sin(t))
        y_next = u * (x * np.sin(t) + y * np.cos(t))
        return x_next, y_next

    x_values = [x0]
    y_values = [y0]
    for _ in range(iterations):
        x, y = ikeda_map(x_values[-1], y_values[-1], u)
        x_values.append(x)
        y_values.append(y)

    return x_values, y_values




#*
@csrf_exempt
def generate_and_save_tinkerbell_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        a = float(body['formData'].get('a' ,0.9))
        b = float(body['formData'].get('b' , -0.6013))
        c = float(body['formData'].get('c' , 2.0))
        d = float(body['formData'].get('d' , 0.50))
        width = int(body['formData'].get('width', 100))
        height = int(body['formData'].get('height', 100))
        iterations = int(body['formData'].get('iterations', 10000))

        # Generate Tinkerbell map
        def tinkerbell_map(x, y):
            xn = x ** 2 - y ** 2 + a * x + b * y
            yn = 2 * x * y + c * x + d * y
            return xn, yn

        tinkerbell_map_array = np.zeros((width, height))

        x, y = 0.1, 0.1
        for _ in range(iterations):
            x, y = tinkerbell_map(x, y)
            ix, iy = int((x + 2) / 4 * width), int((y + 2) / 4 * height)
            if 0 <= ix < width and 0 <= iy < height:
                tinkerbell_map_array[iy, ix] += 1

        # File path
        file_path = os.path.join('chaos_app', 'maps', 'tinkerbell_map.png')

        # If file exists, delete
        if os.path.exists(file_path):
            os.remove(file_path)

        # Plot and save to file
        plt.imshow(tinkerbell_map_array, cmap='hot', origin='lower', extent=(-2, 2, -2, 2))
        plt.title('Tinkerbell Haritası')
        plt.colorbar(label='Ziyaret Sayısı')
        plt.savefig(file_path)
        plt.close()

        # Return URL of saved file
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)
#*
@csrf_exempt
def generate_and_save_logistic_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        r = float(body['formData'].get('r',3.9))
        x0 = float(body['formData'].get('x0',0.5))
        iterations = int(body['formData'].get('iterations',100))

        values = np.zeros(iterations + 1)
        values[0] = x0

        for i in range(iterations):
            values[i + 1] = logistic_map(r, values[i])

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'logistic_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.plot(values, 'b-', lw=0.5)
        plt.title('Logistic Map: r = {}'.format(r))
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)


def logistic_map(r, x):
    return r * x * (1 - x)

#*
@csrf_exempt
def generate_and_save_complex_squaring_map(request):
    if request.method == 'POST':
        # JSON verisini al
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        # Harita oluşturma parametrelerini al
        real_range = float(body['formData'].get('real_range',10))
        imag_range = float(body['formData'].get('imag_range',10))
        num_points = int(body['formData'].get('num_points',100))

        # Karmaşık sayıları oluştur
        real_values = np.linspace(-real_range, real_range, num_points)
        imag_values = np.linspace(-imag_range, imag_range, num_points)
        complex_points = [complex(real, imag) for real in real_values for imag in imag_values]

        # Karmaşık sayıları karesini al
        squared_points = [complex_squaring(z) for z in complex_points]
        real_parts = [z.real for z in squared_points]
        imag_parts = [z.imag for z in squared_points]

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'complex_squaring_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Scatter plot oluşturma ve dosyaya kaydetme
        plt.figure(figsize=(8, 6))
        plt.scatter(real_parts, imag_parts, s=5)
        plt.xlabel('Gerçel Bölüm')
        plt.ylabel('Sanal Bölüm')
        plt.title('Karmaşık Sayı Karesi Haritası')
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)


def complex_squaring(z):
    """Karmaşık sayının karesini alır."""
    real_part = z.real ** 2 - z.imag ** 2
    imag_part = 2 * z.real * z.imag
    return complex(real_part, imag_part)

@csrf_exempt
def arnoldcat_map_api(request):
    if request.method == 'POST':
        # Parse JSON data
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        # Get image data and parameters
        image_data = body.get('image_data')
        iterations = int(body.get('iterations', 1))
        keep_all = bool(body.get('keep_all', False))

        # Save the uploaded image temporarily
        temp_image_path = 'chaos_app/temp_image.png'
        image_content = ContentFile(image_data.encode('utf-8'))
        path = default_storage.save(temp_image_path, image_content)

        # Perform the transformation
        result_image = arnoldcat_map(path, iterations, keep_all)

        # Save the result image
        result_image_path = os.path.join('chaos_app', 'maps', 'ikeda_attractor.png')
        result_image.save(result_image_path)

        # Return the saved file relative path
        plot_url = os.path.join('chaos_app', 'maps', 'ikeda_attractor.png')
        return JsonResponse({'plot_url': plot_url})

    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)
#*
@csrf_exempt
def generate_and_save_bernoulli_map(request):
    if request.method == 'POST':
        # JSON verisini al
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        # Harita oluşturma parametrelerini al
        r = float(body.get('r', 4.6))
        iterations = int(body.get('iterations', 1000))

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'bernoulli_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Bernoulli haritasını oluştur ve dosyaya kaydet
        plot_bernoulli_map(r=r, iterations=iterations, file_path=file_path)

        # Kaydedilen dosyanın göreli URL'sini döndür
        plot_url = os.path.join('chaos_app', 'maps', 'bernoulli_map.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)
#*
@csrf_exempt
def generate_and_save_L96_trajectory(request):
    if request.method == 'POST':
        # JSON verisini al
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        # Parametreleri al
        N = int(body.get('N', 5))
        F = float(body.get('F', 8))
        x0 = body.get('x0', None)
        t_range = body.get('t_range', [0.0, 30.0, 0.01])

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'L96_trajectory.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # L96 modelini çiz ve dosyaya kaydet
        plot_L96_trajectory(N=N, F=F, x0=x0, t_range=t_range, file_path=file_path)

        # Kaydedilen dosyanın göreli dosya yolunu döndür
        plot_url = os.path.join('chaos_app', 'maps', 'L96_trajectory.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)

@csrf_exempt
def lorenz_map(request):
    # Lorenz çekicisini çiz
    image_path = os.path.join(settings.MEDIA_ROOT, 'maps', 'map.png')
    lorenz_plt()
    
    # Grafik dosyasını kaydet
    plt.savefig(image_path)
    plt.close()

    # Oluşturulan dosyanın URL'ini döndür
    image_url = request.build_absolute_uri(settings.MEDIA_URL + 'maps/map.png')
    return JsonResponse({'url': image_url})


@csrf_exempt
def poincare_map(time_series, delay):
    poincare_points = []
    for i in range(len(time_series) - delay):
        poincare_points.append([time_series[i], time_series[i + delay]])
    poincare_points = np.array(poincare_points)
    plt.figure(figsize=(8, 6))
    plt.plot(poincare_points[:, 0], poincare_points[:, 1], 'bo', markersize=2)
    plt.xlabel('x(n)')
    plt.ylabel('x(n + {})'.format(delay))
    plt.title('Poincaré Haritası')
    plt.grid(True)
    return plt

def poincare_map_view(request):
    t = np.arange(0, 100, 0.1)
    time_series = np.sin(t)
    delay = 10  # Gecikme sayısı

    # Poincaré haritasını çiz
    plt = poincare_map(time_series, delay)
    
    # Grafik dosyasını kaydet
    image_path = os.path.join(settings.MEDIA_ROOT, 'maps', 'poincare_map.png')
    plt.savefig(image_path)
    plt.close()

    # Oluşturulan dosyanın URL'ini döndür
    image_url = request.build_absolute_uri(settings.MEDIA_URL + 'maps/poincare_map.png')
    return JsonResponse({'url': image_url})

@csrf_exempt
def generate_and_save_gingerbread_man(request):
    if request.method == 'POST':
        # İstek gövdesini JSON olarak ayrıştır
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        x = float(body['formData'].get('x', 0.1))
        y = float(body['formData'].get('y', -0.1))
        a = float(body['formData'].get('a', 1))
        b = float(body['formData'].get('b', 1))
        iterations = int(body['formData'].get('iterations', 50000))

        # Çizim alanı ve iterasyonlar
        plt.figure(figsize=(8, 6))
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Gingerbread Man Haritası')

        for _ in range(iterations):
            dx = 1 - a * y + b * abs(x)
            dy = x
            x = dx
            y = dy
            plt.plot(x * 10, y * 10, 'b.', markersize=1)

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'gingerbread_man.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)

@csrf_exempt
def generate_and_save_gauss_map(request):
    if request.method == 'POST':
        # JSON isteğini ayrıştır
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        # Parametreleri al
        x0 = float(body['formData'].get('x0'))
        r = float(body['formData'].get('r'))
        iterations = int(body['formData'].get('iterations'))

        # Gauss haritasını oluştur
        sequence = [x0]
        for _ in range(iterations - 1):  # İterasyonları gereksiz hesaplama yapmamak için ayarlayın
            x0 = gauss_map(x0, r)
            sequence.append(x0)

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'gauss_map.png')

        # Dosyayı varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.plot(sequence, 'b-', linewidth=0.5)
        plt.title('Gauss Map: r = {}'.format(r))
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)

def gauss_map(x, r):
    return r * x * (1 - x)


@csrf_exempt
def generate_and_save_lotka_volterra_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        alpha = float(body['formData'].get('alpha'))
        beta = float(body['formData'].get('beta'))
        gamma = float(body['formData'].get('gamma'))
        delta = float(body['formData'].get('delta'))
        prey_initial = float(body['formData'].get('prey_initial'))
        predator_initial = float(body['formData'].get('predator_initial'))

        # Zaman noktaları
        t = np.linspace(0, 100, 1000)

        # Başlangıç koşulları
        y0 = [prey_initial, predator_initial]

        # Lotka-Volterra diferansiyel denklemleri
        def model(y, t):
            prey, predator = y
            dydt = [alpha * prey - beta * prey * predator,
                    gamma * prey * predator - delta * predator]
            return dydt

        # Diferansiyel denklemleri çöz
        sol = odeint(model, y0, t)

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'Lotka-Volterra.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik
        plt.figure(figsize=(10, 6))
        plt.plot(t, sol[:, 0], label='Av Popülasyonu')
        plt.plot(t, sol[:, 1], label='Yırtıcı Popülasyonu')
        plt.xlabel('Zaman')
        plt.ylabel('Popülasyon')
        plt.title('Lotka-Volterra Modeli: Yırtıcı ve Av Popülasyonları')
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)

@csrf_exempt
def generate_and_save_logistic_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        r = float(body['formData'].get('r', 3.9))
        x0 = float(body['formData'].get('x0', 0.5))
        num_steps = int(body['formData'].get('num_steps', 100))

        # Logistic map function
        def logistic_map(r, x0, num_steps):
            x = np.zeros(num_steps)
            x[0] = x0
            for i in range(1, num_steps):
                x[i] = r * x[i - 1] * (1 - x[i - 1])
            return x

        # Calculate logistic map
        population = logistic_map(r, x0, num_steps)

        # File path
        file_path = os.path.join('chaos_app', 'maps', 'logistic_map.png')

        # If file exists, delete
        if os.path.exists(file_path):
            os.remove(file_path)

        # Plot and save to file
        plt.figure(figsize=(8, 8))
        plt.plot(population, 'b-', label='Logistic Haritası')
        plt.title('Logistic Haritası (r={})'.format(r))
        plt.xlabel('Adım')
        plt.ylabel('Popülasyon Oranı')
        plt.legend()
        plt.savefig(file_path)
        plt.close()

        # Return URL of saved file
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)
    
# KAAN KOÇ

def zaslavskii_rotation_map(x, y, a, k):
    x_new = x + y + k * np.sin(2 * np.pi * y)
    y_new = y - a * np.sin(2 * np.pi * x)
    return x_new, y_new
#*
@csrf_exempt
def generate_and_save_zaslavskii_map(request):
    if request.method == 'POST':
        # İstek gövdesini JSON olarak ayrıştırma
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        a = float(body['formData'].get('a', 0.1))
        k = float(body['formData'].get('k', 0.1))
        iterations = int(body['formData'].get('iterations', 10000))

        # Başlangıç koordinatları
        x, y = 0.1, 0.1

        # Değerleri saklamak için dizi
        x_values, y_values = np.zeros(iterations + 1), np.zeros(iterations + 1)
        x_values[0], y_values[0] = x, y

        # İterasyonları yap
        for i in range(1, iterations + 1):
            x, y = zaslavskii_rotation_map(x, y, a, k)
            x_values[i], y_values[i] = x, y

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'zaslavskii_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.figure(figsize=(8, 8))
        plt.plot(x_values, y_values, 'b,', markersize=1)
        plt.title('Zaslavskii Rotation Map')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = os.path.join('chaos_app', 'maps', 'zaslavskii_map.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)
    


def complex_squaring(z):
    real_part = z.real ** 2 - z.imag ** 2
    imag_part = 2 * z.real * z.imag
    return complex(real_part, imag_part)

@csrf_exempt
def generate_and_save_complex_squared_map(request):
    if request.method == 'POST':
        # İstek gövdesini JSON olarak ayrıştırma
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        real_range = float(body['formData'].get('real_range', 10))
        imag_range = float(body['formData'].get('imag_range', 10))
        num_points = int(body['formData'].get('num_points', 100))

        # Karmaşık sayı karesi haritasını oluşturma
        real_values = [i * real_range / num_points for i in range(-num_points, num_points)]
        imag_values = [i * imag_range / num_points for i in range(-num_points, num_points)]

        complex_points = [complex(real, imag) for real in real_values for imag in imag_values]
        squared_points = [complex_squaring(z) for z in complex_points]

        real_parts = [z.real for z in squared_points]
        imag_parts = [z.imag for z in squared_points]

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'complex_squared_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.figure(figsize=(8, 6))
        plt.scatter(real_parts, imag_parts, s=5)
        plt.xlabel('Gerçel Bölüm')
        plt.ylabel('Sanal Bölüm')
        plt.title('Karmaşık Sayı Karesi Haritası')
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)
    
class Agent:
    def __init__(self, length):
        self.params = [random.uniform(1, 4), random.uniform(0.1, 4)]  # (a,b)
        self.fitness = -1

    def __str__(self):
        return 'Params: ' + str(self.params) + ' Fitness: ' + str(self.fitness)

def init_agents(population, length):
    return [Agent(length) for _ in range(population)]

a_vals = []
b_vals = []
f_vals = []
#*
@csrf_exempt
def generate_and_save_genetic_algorithm_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        global population
        global generations
        global plaintext
        population = int(body['formData'].get('population', 20))
        generations = int(body['formData'].get('generations', 100000))
        plaintext = body['formData'].get('plaintext', 'abcdefghij' * 100)

        def ani_jackard(s1, s2):
            str1 = [ord(i) for i in s1]
            str2 = [ord(i) for i in s2]

            str1 = set(str1)
            str2 = set(str2)

            score = (str1 & str2)
            score_u = str1 | str2

            return 100 - (len(score) / len(score_u)) * 100

        def fitness(agents):
            for agent in agents:
                a = agent.params[0]
                d = agent.params[1]

                cipher = encrypt(plaintext, a, d)
                agent.fitness = ani_jackard(plaintext, cipher)

            return agents

        def selection(agents):
            agents = sorted(agents, key=lambda agent: agent.fitness, reverse=True)
            agents = agents[:int(0.2 * len(agents))]
            return agents

        def crossover(agents):
            offspring = []

            for _ in range((population - len(agents)) // 2):
                parent1 = random.choice(agents)
                parent2 = random.choice(agents)
                child1 = Agent(2)
                child2 = Agent(2)
                child1.params = [parent1.params[0], parent2.params[1]]
                child2.params = [parent2.params[0], parent1.params[1]]

                offspring.append(child1)
                offspring.append(child2)

            agents.extend(offspring)
            return agents

        def mutation(agents):
            for agent in agents:
                step_a = random.uniform(-0.2, 0.2)
                step_d = random.uniform(-0.2, 0.2)

                if random.uniform(0.0, 1.0) <= 0.1:
                    agent.params[0] += step_a
                    agent.params[1] += step_d

            return agents

        def chaotic_map(n, x_0, y_0, a, d):
            x = [x_0]
            y = [y_0]

            for i in range(n - 1):
                x.append((x[i] + d + (a * math.sin(2 * math.pi * y[i]))) % 1)
                y.append(1 - a * pow(x[i], 2) + y[i])

            return (x, y)

        def float_to_shuffled_ints(x, y):
            x_sorted = sorted(x, reverse=True)
            y_sorted = sorted(y, reverse=True)

            shuffled_x = [x_sorted.index(x_val) for x_val in x]
            shuffled_y = [y_sorted.index(y_val) for y_val in y]

            key = [shuffled_y[i] for i in shuffled_x]
            return key

        def encrypt(plaintext, a, d):
            ascii_lst = [ord(i) for i in plaintext]
            n = len(ascii_lst)

            ascii_avg = sum(ascii_lst) / n
            x_0 = ascii_avg / max(ascii_lst)
            y_0 = 0.2
            (x, y) = chaotic_map(n, x_0, y_0, a, d)

            private_key = float_to_shuffled_ints(x, y)

            ciphertext = [chr(ascii_lst[i] + private_key[i]) for i in range(len(ascii_lst))]
            return ''.join(ciphertext)

        def ga():
            global a_vals, b_vals, f_vals
            a_vals = []
            b_vals = []
            f_vals = []
            agents = init_agents(population, 2)

            for generation in range(generations):
                agents = fitness(agents)
                temp_fitness = [agent.fitness for agent in agents]

                current_max_fitness = max(temp_fitness)
                count = temp_fitness.count(current_max_fitness)

                if count / len(agents) >= 0.5 and current_max_fitness >= 90:
                    break

                for agent in agents:
                    a_vals.append(agent.params[0])
                    b_vals.append(agent.params[1])
                    f_vals.append(agent.fitness)

                agents = selection(agents)
                agents = crossover(agents)
                agents = mutation(agents)

        # Run the genetic algorithm
        ga()

        # File path
        file_path = os.path.join('chaos_app', 'maps', 'genetic_algorithm_map.png')

        # If file exists, delete
        if os.path.exists(file_path):
            os.remove(file_path)

        # Plot and save to file
        xlist = np.array(a_vals)
        ylist = np.array(b_vals)
        zlist = np.array(f_vals)

        plt.xlabel('Variation of a')
        plt.ylabel('Variation of b')
        plt.title('Length of Plaintext: {}'.format(len(plaintext)))
        scatter = plt.scatter(xlist, ylist, c=zlist)
        cbar = plt.colorbar(scatter)
        cbar.set_label("Fitness")
        plt.savefig(file_path)
        plt.close()

        # Return URL of saved file
        plot_url = os.path.join('chaos_app', 'maps', 'genetic_algorithm_map.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)
        



# bifurkasyon.py içeriği

import numpy as np
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
#*
@csrf_exempt
def generate_and_save_bifurcation(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        r_min = float(body['formData'].get('r_min', 2.5))
        r_max = float(body['formData'].get('r_max', 4.0))
        num_r = int(body['formData'].get('num_r', 1000))
        iterations = int(body['formData'].get('iterations', 1000))
        last = int(body['formData'].get('last', 100))

        r_values = np.linspace(r_min, r_max, num_r)
        x = 1e-5 * np.ones(num_r)

        result = []
        for _ in range(iterations):
            x = r_values * x * (1 - x)
            if _ >= (iterations - last):
                result.append(np.copy(x))

        x_vals = np.array(result).T

        file_path = os.path.join('chaos_app', 'maps', 'bifurcation_map.png')
        if os.path.exists(file_path):
            os.remove(file_path)

        plt.figure(figsize=(10, 6))
        plt.plot(r_values, x_vals, ',k', alpha=0.25)
        plt.title('Bifurcation Diagram')
        plt.xlabel('r')
        plt.ylabel('x')
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        plot_url = os.path.join('chaos_app', 'maps', 'bifurcation_map.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)

# lorenz_cekici.py içeriği

import numpy as np
import matplotlib.pyplot as plt
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import os
from scipy.integrate import odeint
#*
@csrf_exempt
def generate_and_save_lorenz(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        sigma = float(body['formData'].get('sigma', 10.0))
        beta = float(body['formData'].get('beta', 2.667))
        rho = float(body['formData'].get('rho', 28.0))
        num_steps = int(body['formData'].get('num_steps', 10000))
        dt = float(body['formData'].get('dt', 0.01))

        def lorenz(X, t, sigma, beta, rho):
            x, y, z = X
            dx_dt = sigma * (y - x)
            dy_dt = x * (rho - z) - y
            dz_dt = x * y - beta * z
            return [dx_dt, dy_dt, dz_dt]

        t = np.linspace(0, num_steps*dt, num_steps)
        X0 = [0.0, 1.0, 1.05]
        X = odeint(lorenz, X0, t, args=(sigma, beta, rho))

        file_path = os.path.join('chaos_app', 'maps', 'lorenz_attractor.png')
        if os.path.exists(file_path):
            os.remove(file_path)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:,0], X[:,1], X[:,2], lw=0.5)
        ax.set_title('Lorenz Attractor')
        plt.savefig(file_path)
        plt.close()

        plot_url = os.path.join('chaos_app', 'maps', 'lorenz_attractor.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)

#poinecare_map içeriği
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import matplotlib.pyplot as plt
import os
import json
#*
@csrf_exempt
def generate_poincare_map(request):
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        omega = float(body['formData'].get('omega', 1.0))
        F = float(body['formData'].get('F', 0.5))
        phi = float(body['formData'].get('phi', 0.0))
        t_max = float(body['formData'].get('t_max', 100.0))
        dt = float(body['formData'].get('dt', 0.01))

        def poincare_map(t, x):
            return x[1], F * np.sin(omega * t + phi) - 0.2 * x[1] - x[0]

        t_values = np.arange(0, t_max, dt)
        x = np.zeros((len(t_values), 2))
        for i, t in enumerate(t_values[:-1]):
            k1 = np.array(poincare_map(t, x[i]))
            k2 = np.array(poincare_map(t + dt / 2, x[i] + dt / 2 * k1))
            k3 = np.array(poincare_map(t + dt / 2, x[i] + dt / 2 * k2))
            k4 = np.array(poincare_map(t + dt, x[i] + dt * k3))
            x[i + 1] = x[i] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        plt.figure(figsize=(10, 6))
        plt.plot(x[:, 0], x[:, 1], 'b-', linewidth=0.5, marker='o', markersize=2)
        plt.title('Poincaré Map')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)

        file_path = os.path.join('chaos_app', 'maps', 'poincare_map.png')
        if os.path.exists(file_path):
            os.remove(file_path)

        plt.savefig(file_path)
        plt.close()
        
        plot_url = os.path.join('chaos_app', 'maps', 'poincare_map.png')
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)
