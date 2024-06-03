from django.db import models
import numpy as np
import os
from PIL.Image import open as load_pic, new as new_pic
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def L96(x, t, N, F):
    d = np.zeros(N)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return d

def plot_L96_trajectory(N=5, F=8, x0=None, t_range=(0.0, 30.0, 0.01), file_path='chaos_app/maps/L96_trajectory.png'):
    if x0 is None:
        x0 = F * np.ones(N)
        x0[0] += 0.01

    t = np.arange(t_range[0], t_range[1], t_range[2])  # Create time array

    # Solve ODE using odeint
    x = integrate.odeint(L96, x0, t, args=(N, F))

    # Create and save the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(x[:, 0], x[:, 1], x[:, 2])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    plt.savefig(file_path)
    plt.close()


def arnoldcat_map(path, iterations, keep_all=False, name="arnold_cat-{name}-{index}.png"):
    title = os.path.splitext(os.path.split(path)[1])[0]
    counter = 0
    while counter < iterations:
        with load_pic(path) as image:
            dim = width, height = image.size
            with new_pic(image.mode, dim) as canvas:
                for x in range(width):
                    for y in range(height):
                        nx = (2 * x + y) % width
                        ny = (x + y) % height
                        canvas.putpixel((nx, height - ny - 1), image.getpixel((x, height - y - 1)))

        if counter > 0 and not keep_all:
            os.remove(path)
        counter += 1
        path = name.format(name=title, index=counter)
        canvas.save(path)

    return canvas

def bernoulli_map(x, r):
    return r * x * (1 - x)

def plot_bernoulli_map(r=4.6, iterations=1000, file_path='chaos_app/maps/bernoulli_map.png'):
    x = np.linspace(0, 1, 1000)
    y = bernoulli_map(x, r)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'r-', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(f'Bernoulli Haritası (r={r})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(file_path)
    plt.close()