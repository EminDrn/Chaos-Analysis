from flask import Flask, send_file
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Başlangıç noktası ve kaos parametresi
x0 = 0.2
r = 3.9
# Kaç adım yapılacağı
n = 10000

# Gauss haritası oluşturma fonksiyonu
def gauss_map(x, r):
    return r * x * (1 - x)

def generate_map(x0, r, n):
    map_values = [x0]
    for _ in range(n):
        x0 = gauss_map(x0, r)
        map_values.append(x0)
    plt.figure()
    plt.plot(map_values[:-1], map_values[1:], '.', markersize=1)
    plt.xlabel('x_n')
    plt.ylabel('x_{n+1}')
    plt.title(f"Gauss Map (r={r})")
    plt.savefig('maps/gauss_map.png')

# API endpoint'i
@app.route('/generate_map')
def generate_and_save_map():
    generate_map(x0, r, n)
    return send_file('maps/gauss_map.png', mimetype='image/png')

if __name__ == '__main__':
    # maps klasörünü oluştur
    if not os.path.exists('maps'):
        os.makedirs('maps')
    app.run(debug=True)