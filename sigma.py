import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
z = 2/(1 + np.exp(-x))-1

plt.plot(x, z)
plt.savefig("foo.png")
