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
#model import
from .models import lorenz96_model,bernoulli_map


@csrf_exempt
def generate_and_save_tent_map(request):
    if request.method == 'POST':
        # Parse request body as JSON

        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        x = float(body['formData'].get('x'))
        r = float(body['formData'].get('r'))
        iterations = int(body['formData'].get('iterations'))

        sequence = [x]
        for _ in range(iterations - 1):  # Adjust iterations to avoid redundant calculation
            x = tent_map(x, r)
            sequence.append(x)

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'tent_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.plot(sequence, 'b-', linewidth=0.5)
        plt.title('Tent Map: r = {}'.format(r))
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)


def tent_map(x, r):
    if x < 0.5:
        return r * x
    else:
        return r * (1 - x)

@csrf_exempt
def generate_and_save_tinkerbell_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        a = float(body['formData'].get('a'))
        b = float(body['formData'].get('b'))
        c = float(body['formData'].get('c'))
        d = float(body['formData'].get('d'))
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

@csrf_exempt
def generate_and_save_logistic_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        r = float(body['formData'].get('r'))
        x0 = float(body['formData'].get('x0'))
        iterations = int(body['formData'].get('iterations'))

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


@csrf_exempt
def generate_and_save_complex_squaring_map(request):
    if request.method == 'POST':
        # JSON verisini al
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        # Harita oluşturma parametrelerini al
        real_range = float(body['formData'].get('real_range'))
        imag_range = float(body['formData'].get('imag_range'))
        num_points = int(body['formData'].get('num_points'))

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
def generate_and_save_bernoulli_map(request):
    if request.method == 'POST':
        # JSON verisini al
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        # Harita oluşturma parametrelerini al
        r = float(body['formData'].get('r'))
        x0 = float(body['formData'].get('x0'))
        num_iterations = int(body['formData'].get('num_iterations'))

        # Bernoulli haritasını oluştur
        x_values = [x0]
        for _ in range(num_iterations):
            x_next = bernoulli_map(x_values[-1], r)
            x_values.append(x_next)

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'bernoulli_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.plot(x_values, 'b-', lw=0.5)
        plt.title('Bernoulli Map: r = {}'.format(r))
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)


@csrf_exempt
def generate_and_save_lorenz96_map(request):
    if request.method == 'POST':
        # JSON verisini al
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        # Model parametrelerini al
        N = int(body['formData'].get('N', 40))  # Varsayılan olarak 40
        F = float(body['formData'].get('F', 8.0))  # Varsayılan olarak 8.0
        dt = float(body['formData'].get('dt', 0.01))  # Varsayılan olarak 0.01
        num_steps = int(body['formData'].get('num_steps', 1000))  # Varsayılan olarak 1000

        # Modeli çalıştır ve verileri sakla
        data = lorenz96_model(N, F, dt, num_steps)

        # Dosya yolu
        file_path = os.path.join('chaos_app', 'maps', 'lorenz96_map.png')

        # Eğer dosya varsa sil
        if os.path.exists(file_path):
            os.remove(file_path)

        # Grafik çiz ve dosyaya kaydet
        plt.imshow(data, aspect='auto', cmap='jet')
        plt.colorbar()
        plt.title('Lorenz 96 Map')
        plt.xlabel('Time Step')
        plt.ylabel('Variable Index')
        plt.savefig(file_path)
        plt.close()

        # Kaydedilen dosyanın URL'sini döndür
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': file_path})
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
def generate_and_save_gingerbread_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        iterations = int(body['formData'].get('iterations', 10000))
        x_start = float(body['formData'].get('x_start', 0.1))
        y_start = float(body['formData'].get('y_start', 0.1))

        # Gingerbread map function
        def gingerbread_map(x, y):
            x_next = 1 - y + abs(x)
            y_next = x
            return x_next, y_next

        x = x_start
        y = y_start

        x_points = [x]
        y_points = [y]

        for _ in range(iterations):
            x, y = gingerbread_map(x, y)
            x_points.append(x)
            y_points.append(y)

        # File path
        file_path = os.path.join('chaos_app', 'maps', 'gingerbread_map.png')

        # If file exists, delete
        if os.path.exists(file_path):
            os.remove(file_path)

        # Plot and save to file
        plt.figure(figsize=(8, 8))
        plt.plot(x_points, y_points, 'o-', markersize=1, alpha=0.5)
        plt.title('Gingerbread Man Haritası')
        plt.savefig(file_path)
        plt.close()

        # Return URL of saved file
        return JsonResponse({'plot_url': file_path})
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
        plot_url = request.build_absolute_uri(file_path)
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
        plot_url = request.build_absolute_uri(file_path)
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)