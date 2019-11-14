import matplotlib.pyplot as plt
import numpy as np

residuals = np.loadtxt('cg_residuals.csv', delimiter=',', skiprows=1)
plt.plot(residuals[:,0], residuals[:,1])
plt.yscale('log')
plt.show()
