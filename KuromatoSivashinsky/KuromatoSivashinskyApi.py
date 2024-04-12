from flask import Flask, jsonify
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

app = Flask(__name__)

def kuramoto_sivashinsky(t, u, alpha, beta, L):
    du_dt = -alpha * np.fft.fftfreq(len(u))**2 * np.fft.fft(u) - beta * 0.5j * np.fft.fftn(u)**2
    return np.real(np.fft.ifft(du_dt))

def solve_kuramoto_sivashinsky(alpha, beta, L, T, N):
    x = np.linspace(0, L, N)
    u0 = np.sin(np.pi * x / L) + 0.5 * np.sin(2 * np.pi * x / L)
    sol = solve_ivp(kuramoto_sivashinsky, [0, T], u0, args=(alpha, beta, L), t_eval=np.linspace(0, T, 100))
    return sol.t, sol.y.T

def plot_solution(alpha, beta, L, T, N):
    t, u = solve_kuramoto_sivashinsky(alpha, beta, L, T, N)
    x = np.linspace(0, L, len(u[0]))
    plt.figure()
    for i in range(0, len(t), len(t) // 10):
        plt.plot(x, u[i], label='t = {:.2f}'.format(t[i]))
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Kuramoto-Sivashinsky Denklemi Modellemesi')
    plt.legend()
    plt.tight_layout()
    plt.savefig('maps/kuramoto_sivashinsky.png')
    plt.close()

@app.route('/generate_image/<float:alpha>/<float:beta>/<float:L>/<float:T>/<int:N>', methods=['GET'])
def generate_image(alpha, beta, L, T, N):
    plot_solution(alpha, beta, L, T, N)
    return jsonify({"image_url": "maps/kuramoto_sivashinsky.png"})

if __name__ == '__main__':
    app.run(debug=True)