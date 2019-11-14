import matplotlib
import matplotlib.pyplot as plt
import numpy as np


font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

titles = [
        'No Randomization without AMG Preconditioning',
        'No Randomization with AMG Preconditioning',
        'RHS Randomization without AMG Preconditioning',
        'RHS Randomization with AMG Preconditioning',
        'RHS and Boundary Randomization without AMG Preconditioning',
        'RHS and Boundary Randomization with AMG Preconditioning',
        ]

suptitle = 'Effect of AMG Preconditioning on CG Convergence of RHS-Randomized 2D Poisson Problems'

sample_start = 2
sample_end = 4
rows = 1
cols = 2

fig = plt.figure()
fig.suptitle(suptitle)
plt.subplots_adjust(hspace=0.4)

for i in range(sample_start + 1, sample_end + 1):
    residuals = np.loadtxt(f'poisson-{i}.csv', delimiter=',', skiprows=1)
    ax = fig.add_subplot(rows, cols, i - sample_start)
    ax.plot(residuals[:,0], residuals[:,1])
    ax.set_yscale('log')
    ax.set_xlabel('CG Iterations')
    ax.set_ylabel('Residual')
    ax.set_title(titles[i-1])

plt.show()
