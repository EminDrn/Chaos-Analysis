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