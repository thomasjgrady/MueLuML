import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)

plt.figure()
ax = plt.gca()
ax.bar([1, 2], [48, 9])
ax.set_xticks([1, 2])
ax.set_xticklabels(['Default Parameters', 'Neural Network'])

plt.ylabel('Number of CG Iterations')
plt.title('Parameter Selection Method vs Number of CG Iterations for Apache1 Matrix')
plt.show()
