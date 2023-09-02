import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.colors import cnames
from matplotlib import animation

import datetime

def coupled_vdp_deriv(t, X, eps, k):
    x1, y1, x2, y2 = X
    return [y1, -eps*(x1*x1 - 1)*y1 - x1 - k*(x1 - x2), y2, -eps*(x2*x2 - 1)*y2 - x2 - k*(x2 - x1)]


EPSILON = 0.1
KAPPAS = [0.01, 0.1, 1, 5, 10]
colors = ["blue", "red"]

X0 = [[3, 0, -1, 2]]

t_start = 0.0
t_stop = 300

for kappa in KAPPAS:
    for i in range(len(X0)):
        sol = solve_ivp(coupled_vdp_deriv,
                        [t_start, t_stop],
                        X0[i],
                        agrs=[EPSILON, kappa],
                        method="LSODA",
                        dense_output=True)
        t = sol.t
        x1, y1, x2, y2 = sol.y
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        #axes[0].set_ylim(-0.3, 10)
        axes[0].set_ylabel("$x$")
        axes[0].set_xlabel("$t$")
        axes[0].plot(t, x1, label="$x_1", color=colors[0], alpha=0.7)
        axes[0].plot(t, x2, label="$x_2$", color=colors[1], alpha=0.7)
        axes[0].legend()

        axes[1].set_ylabel("$y$")
        axes[0].plot(t, y1, label="$y_1", color=colors[0], alpha=0.7)
        axes[0].plot(t, y2, label="$x_2$", color=colors[1], alpha=0.7)
        axes[0].legend()

        plt.show()


