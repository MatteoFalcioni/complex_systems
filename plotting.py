import matplotlib.pyplot as plt
import statistics as stat
import numpy as np
import math

with open('sim.data') as f1:
    lines = f1.readlines()
    x1 = [int(line.split()[0]) for line in lines]
    t1 = [float(line.split()[1]) for line in lines]

with open('sim_pert.data') as f2:
    lines = f2.readlines()
    x2 = [int(line.split()[0]) for line in lines]
    t2 = [float(line.split()[1]) for line in lines]

t_unperturbed = [value for value in t1 if value > 0]
t_perturbed = [value for value in t2 if value > 0]
mean_t_pert = stat.mean(t_perturbed)
mean_t_unpert = stat.mean(t_unperturbed)
sigma1 = math.sqrt(np.var(t_unperturbed, ddof=0))
sigma2 = math.sqrt(np.var(t_perturbed, ddof=0))
print(str(sigma1) + " " + str(sigma2))

t_pert_dev = stat.stdev(t_perturbed)
t_unpert_dev = stat.stdev(t_unperturbed)
print("avg value of unperturbed times is t_unp = " + str(mean_t_unpert) + " +/- " + str(t_unpert_dev))
print("avg value of perturbed times is t_pert = " + str(mean_t_pert) + " +/- " + str(t_pert_dev))

n = len(x1)
fig1, ax1 = plt.subplots(1)
fig1.suptitle('probability of finding food')
p_unperturbed = x1.count(1)/n
p_perturbed = x2.count(1)/n
width = 0.1
ax1.bar("unperturbed system", p_unperturbed, width, color="green")
ax1.bar("perturbed system", p_perturbed, width, color="red")
margin = (1 - width) + width / 2
ax1.set_xlim(-margin, 2 - 1 + margin)
ax1.set_ylim(0, 1)
plt.show()

n_bins = 30
fig2 = plt.figure()
gs = fig2.add_gridspec(2, hspace=0.5)
ax2 = gs.subplots(sharex=True, sharey=True)
fig2.suptitle('distribution of number of iterations needed to find food')
ax2[0].hist(t_unperturbed, n_bins)
ax2[0].axvline(stat.mean(t_unperturbed), color='k', linestyle='dashed', linewidth=1)
ax2[0].set_title('unperturbed system')
ax2[1].hist(t_perturbed, n_bins)
ax2[1].set_title('perturbed system')
ax2[1].axvline(stat.mean(t_perturbed), color='k', linestyle='dashed', linewidth=1)
for ax in ax2.flat:
    ax.set(xlabel='number of iterations', ylabel='occurrences')
    ax.label_outer()
plt.show()

# I want to plot probability of finding food vs number of ants left after predator appearance for perturbed system




