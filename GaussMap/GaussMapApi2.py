from flask import Flask, send_file
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

class MapGenerator:
    def __init__(self, x0, r, n):
        self.x0 = x0
        self.r = r
        self.n = n

    def gauss_map(self, x, r):
        return r * x * (1 - x)

    def generate_map(self):
        map_values = [self.x0]
        for _ in range(self.n):
            self.x0 = self.gauss_map(self.x0, self.r)
            map_values.append(self.x0)
        plt.figure()
        plt.plot(map_values[:-1], map_values[1:], '.', markersize=1)
        plt.xlabel('x_n')
        plt.ylabel('x_{n+1}')
        plt.title(f"Gauss Map (r={self.r})")
        plt.savefig('maps/gauss_map.png')

@app.route('/generate_map')
def generate_and_save_map():
    map_generator = MapGenerator(x0=0.2, r=3.9, n=10000)
    map_generator.generate_map()
    return send_file('maps/gauss_map.png', mimetype='image/png')

if __name__ == '__main__':
    if not os.path.exists('maps'):
        os.makedirs('maps')
    app.run(debug=True)