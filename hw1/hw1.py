import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, jsonify
from io import BytesIO
import matplotlib

# 使用非GUI後端
matplotlib.use('Agg')

app = Flask(__name__)

def generate_random_points(a, b, c, variance, n=100):
    x = np.linspace(-10, 10, n)
    noise = np.random.normal(0, variance, n)
    y = a * x + b + c * noise
    return x, y

@app.route('/', methods=['GET', 'POST'])
def index():
    a = request.args.get('a', default=1, type=float)
    b = request.args.get('b', default=50, type=float)
    c = request.args.get('c', default=1, type=float)
    variance = request.args.get('variance', default=1, type=float)
    
    return render_template('index.html', a=a, b=b, c=c, variance=variance)

@app.route('/plot')
def plot():
    a = request.args.get('a', default=1, type=float)
    b = request.args.get('b', default=50, type=float)
    c = request.args.get('c', default=1, type=float)
    variance = request.args.get('variance', default=1, type=float)
    
    x, y = generate_random_points(a, b, c, variance)
    
    plt.figure()
    plt.scatter(x, y, color='blue')
    plt.plot(x, a*x + b, color='red')
    plt.title(f'Linear Regression: y = {a}*x + {b} + {c}*N(0, {variance})')
    plt.xlabel('x')
    plt.ylabel('y')
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    
    return send_file(img, mimetype='image/png')

@app.route('/update', methods=['GET'])
def update():
    a = request.args.get('a', default=1, type=float)
    b = request.args.get('b', default=50, type=float)
    c = request.args.get('c', default=1, type=float)
    variance = request.args.get('variance', default=1, type=float)

    x, y = generate_random_points(a, b, c, variance)
    data = {
        'a': a,
        'b': b,
        'c': c,
        'variance': variance
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
