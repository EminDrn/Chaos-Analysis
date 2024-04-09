import json

import numpy as np
import matplotlib.pyplot as plt
import os
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse


@csrf_exempt
def generate_and_save_tent_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        x = float(body.get('x'))
        r = float(body.get('r'))
        iterations = int(body.get('iterations'))

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
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)


def tent_map(x, r):
    if x < 0.5:
        return r * x
    else:
        return r * (1 - x)


@csrf_exempt
def generate_and_save_logistic_map(request):
    if request.method == 'POST':
        # Parse request body as JSON
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)

        r = float(body.get('r'))
        x0 = float(body.get('x0'))
        iterations = int(body.get('iterations'))

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
        return JsonResponse({'plot_url': plot_url})
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
        real_range = float(body.get('real_range'))
        imag_range = float(body.get('imag_range'))
        num_points = int(body.get('num_points'))

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
        return JsonResponse({'plot_url': plot_url})
    else:
        return JsonResponse({'error': 'Only POST requests are supported for this endpoint.'}, status=400)


def complex_squaring(z):
    """Karmaşık sayının karesini alır."""
    real_part = z.real ** 2 - z.imag ** 2
    imag_part = 2 * z.real * z.imag
    return complex(real_part, imag_part)
