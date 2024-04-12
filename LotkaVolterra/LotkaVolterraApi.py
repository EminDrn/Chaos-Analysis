from flask import Flask, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os

app = Flask(__name__)


def lotka_volterra(alpha, beta, gamma, delta, prey_initial, predator_initial, filename):
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

    # Grafik
    plt.figure(figsize=(10, 6))
    plt.plot(t, sol[:, 0], label='Av Popülasyonu')
    plt.plot(t, sol[:, 1], label='Yırtıcı Popülasyonu')
    plt.xlabel('Zaman')
    plt.ylabel('Popülasyon')
    plt.title('Lotka-Volterra Modeli: Yırtıcı ve Av Popülasyonları')
    plt.legend()
    plt.grid(True)

    # Klasöre kaydet
    img_path = os.path.join("maps", filename)
    plt.savefig(img_path)
    plt.close()
    return img_path


@app.route('/lotka_volterra', methods=['POST'])
def calculate_lotka_volterra():
    data = request.get_json()
    alpha = float(data['alpha'])
    beta = float(data['beta'])
    gamma = float(data['gamma'])
    delta = float(data['delta'])
    prey_initial = float(data['prey_initial'])
    predator_initial = float(data['predator_initial'])
    filename = "lotka_volterra.png"  # Aynı dosya adını kullan

    img_path = lotka_volterra(alpha, beta, gamma, delta, prey_initial, predator_initial, filename)
    return jsonify({"image_url": img_path})


if __name__ == '__main__':
    app.run(debug=True)