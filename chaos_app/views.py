import numpy as np
import matplotlib.pyplot as plt
import os

from django.http import JsonResponse

def generate_and_save_tent_map(request):
    x = float(request.GET.get('x'))
    r = float(request.GET.get('r'))
    iterations = int(request.GET.get('iterations'))

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

def tent_map(x, r):
    if x < 0.5:
        return r * x
    else:
        return r * (1 - x)

