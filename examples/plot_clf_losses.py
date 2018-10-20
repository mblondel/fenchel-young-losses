# Author: Mathieu Blondel
# License: Simplified BSD

import os, sys
currentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(currentdir))

import numpy as np
import matplotlib.pylab as plt
import matplotlib

from fyl_numpy import TsallisLoss


matplotlib.rcParams['font.size'] = 13

colors = plt.cm.tab10.colors

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))

s_vals = np.linspace(-3, 3, 100)
p_vals = np.linspace(1e-9, 1-1e-9, 100)
y_true = np.array([1, 0])

i = 0
for name, alpha in (("logistic", 1.0),
                    ("Tsallis", 1.5),
                    ("sparsemax", 2.0),
                    ("perceptron", np.inf)):

    loss_func = TsallisLoss(alpha=alpha)

    name += r" ($\alpha=%s$)" % str(alpha)

    ax3.plot(s_vals, [loss_func([y_true], [[s, 0]]) for s in s_vals],
            label=name, lw=2, color=colors[i])

    ax2.plot(s_vals, [loss_func.predict([[s, 0]])[0, 0] for s in s_vals],
            label=name, lw=2, color=colors[i])

    ax1.plot(p_vals, [-loss_func.Omega([[p, 1-p]])[0] for p in p_vals],
            label=name, lw=2, color=colors[i])

    i += 1

ax3.set_title(r'Loss $L_{\Omega}([s, 0]; e_1)$')
ax3.set_xlabel('s')

ax2.set_title(r'Predictive distribution $\widehat{y}_{\Omega}([s, 0])_1$')
ax2.set_xlabel('s')

ax1.set_title(r'Entropy $-\Omega([p, 1-p])$')
ax1.set_xlabel('p')

ax3.legend(loc="best")

plt.subplots_adjust(bottom=.2)

plt.tight_layout()

plt.show()
