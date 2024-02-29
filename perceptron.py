import matplotlib.pyplot as plt
import numpy as np
import imageio

x = np.array([[1, -1, -1], [1, 1, -1], [1, -1, 1], [1, 1, 1]])
y = np.array([-1, 1, 1, 1])
w = np.array([0, -2, 1])
alpha = 0.1

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

frames = []

plt.figure(figsize=(10, 10), dpi=100)
plt.grid(True)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title("Perceptron")
scatter = plt.scatter(x[:, 1], x[:, 2], c=y)
line, = plt.plot(x[:, 1], -(w[0] + w[1] * x[:, 1]) / w[2])

epoch = 7
for i in range(epoch):
    for j in range(4):
        y_hot = sign(np.dot(w, x[j]))
        Error = y[j] - y_hot
        scatter.set_offsets(x[:, 1:])
        line.set_ydata(-(w[0] + w[1] * x[:, 1]) / w[2])
        plt.draw()
        plt.pause(0.5)
        canvas = plt.gcf().canvas
        canvas.draw()
        buf = canvas.tostring_rgb()
        ncols, nrows = canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        frames.append(img)
        
        print(f"Epoch {i} , k = {j} , Error = {Error} , w = {w} , x = {x[j]} , y = {y[j]} , y_hot = {y_hot}")
        if Error != 0:
            w = w + alpha *(y[j] - y_hot) * x[j]

imageio.mimsave('perceptron.gif', frames)

plt.show()
