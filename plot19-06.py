import numpy as np
import matplotlib.pyplot as plt


w0 = 1
gamma = 0.1

W = np.linspace(0, 5, 1000)


Rho2 = 1/((w0**2 - W**2)**2 + W**2*gamma)

plt.plot(W, Rho2)
plt.show()




