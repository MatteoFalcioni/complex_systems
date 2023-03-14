import matplotlib.pyplot as plt
import statistics as stat
import numpy as np
import math
import random
import ants as a

"""

with open('sim.data') as f1:
    lines = f1.readlines()
    x1 = [int(line.split()[0]) for line in lines]
    t1 = [float(line.split()[1]) for line in lines]

n_sim = len(x1)
ants_left_vs_probability = [0.0]*56
counter = [0]*56

with open('sim_pert.data') as f2:
    lines = f2.readlines()
    x2 = [int(line.split()[0]) for line in lines]
    t2 = [float(line.split()[1]) for line in lines]
    ants_left = [int(line.split()[2]) for line in lines]
    for line in lines:
        if int(line.split()[0]) == 1:
            ants_left_vs_probability[int(line.split()[2])] += 1  # wrong
        counter[int(line.split()[2])] += 1  # counts occurrence of value

ants_vs_prob_norm = []
for i in range(len(ants_left_vs_probability)):
    if counter[i] != 0:
        ants_left_vs_probability[i] = ants_left_vs_probability[i]/counter[i]

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

fig1, ax1 = plt.subplots(1)
fig1.suptitle('probability of finding food')
p_unperturbed = x1.count(1)/n_sim
p_perturbed = x2.count(1)/n_sim
print("p_unperturbed = " + str(p_unperturbed))
print("p_perturbed = " + str(p_perturbed))
width = 0.1
ax1.bar("unperturbed system", p_unperturbed, width, color="green")
ax1.bar("perturbed system", p_perturbed, width, color="red")
margin = (1 - width) + width / 2
ax1.set_xlim(-margin, 2 - 1 + margin)
ax1.set_ylim(0, 1.1)
plt.ylabel("p")
plt.show()

n_bins = 40
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

fig3, ax3 = plt.subplots(1)
fig3.suptitle('probability of finding food vs number of ants left after predation')

labels = [0]*56
for i in range(56):
    labels[i] = i
ax3.scatter(labels, ants_left_vs_probability, marker=".", color="blue", s=20, zorder=5)
ax3.plot(labels, ants_left_vs_probability, color="lightskyblue", zorder=0)
ax3.set_xlim(0, 56)
ax3.set_ylim(0, 1.1)
ax3.set_xticks([0, 10, 20, 30, 40, 50, 55])
plt.ylabel("p")
plt.xlabel("number of ants left")
plt.show()

# let's plot a dummy network for the report
colony = []
N = 50
K = 55
predator_pos = a.Pos(0.8, 0.8)
for i in range(N):  # filling colony
    position = predator_pos
    T_h = random.randint(a.min_t_h, a.max_t_h)  # generate t_h
    C = random.uniform(a.min_c, a.max_c)  # generate c
    while predator_pos.x - a.predator_rng < position.x < predator_pos.x + a.predator_rng or predator_pos.y - a.predator_rng < position.y < predator_pos.y + a.predator_rng:
        position = a.pos_generator(0.7, a.p_nest)
    ant_gen = a.Ant(a.s_0, position, T_h, C, a.v_search)  # generate ants around the nest
    colony.append(ant_gen)
colony.append(a.Ant(0, a.p_nest, 0, 0, 0))
for i in range(5):
    T_h = random.randint(a.min_t_h, a.max_t_h)  # generate t_h
    C = random.uniform(a.min_c, a.max_c)  # generate c
    colony.append(a.Ant(a.s_0, a.pos_generator(0.03, a.p_nest), T_h, C, a.v_search))

fig, ax = plt.subplots(1)
fig.suptitle('predation response')
colony[0].z = a.Pos(0.7, 0.65)
colony[1].z = a.Pos(0.68, 0.65)
colony[2].z = a.Pos(0.85, 0.82)
colony[3].z = a.Pos(0.72, 0.75)

A = [[0 for j in range(K)] for i in range(K)]  # Adjacency matrix
for i in range(K):  # adjourning adjacency matrix at each time step
    for j in range(K):
        if j < i:  # undirected graph -> symmetric adjacency matrix -> j<i is enough
            if a.dist(colony[i].z, colony[j].z) <= a.D:
                A[i][j] = 1
                A[j][i] = 1  # symmetric
            else:
                A[i][j] = 0
                A[j][i] = 0

for i in range(K):
    colony[i].alarm = True
for i in range(K):
    if a.dist(colony[i].z, predator_pos) < a.predator_rng:
        colony[i].alive = False
for i in range(K):
    if a.dist(colony[i].z, a.p_nest) < a.max_c:
        colony[i].safe = True

for i in range(K):
    if colony[i].alive:
        if colony[i].alarm:
            if not colony[i].safe:
                ax.scatter(colony[i].z.x, colony[i].z.y, marker=".", s=85, color="red", zorder=5, label="alarmed ants")
            else:
                ax.scatter(colony[i].z.x, colony[i].z.y, marker=".", s=85, color="blue", zorder=5, label="safe ants")
        else:
            ax.scatter(colony[i].z.x, colony[i].z.y, marker=".", s=85, color="green", zorder=5, label="unalarmed ants")
        for j in range(K + 1):
            if j < i and A[i][j] == 1 and colony[j].alive and not colony[i].safe and not colony[j].safe:
                x_values = [colony[i].z.x, colony[j].z.x]
                y_values = [colony[i].z.y, colony[j].z.y]
                ax.plot(x_values, y_values, color="lightskyblue", zorder=0)
    else:
        ax.scatter(colony[i].z.x, colony[i].z.y, marker="x", s=55, color="black", label="dead ants")


ax.add_patch(plt.Circle((predator_pos.x, predator_pos.y), a.predator_rng, color='orange', fill=False))
ax.add_patch(plt.Circle((a.p_nest.x, a.p_nest.y), a.max_c, color='blue', fill=False))
ax.scatter(predator_pos.x, predator_pos.y, marker="X", s=80, color="orange", zorder=5, label="predator")
ax.scatter(a.p_nest.x, a.p_nest.y, marker="*", s=80, color="blue", zorder=5, label="nest")
ax.set(xlabel='x(t)', ylabel='y(t)')
plt.title('ants network')
plt.show()

"""
# t_h = t_tired
# old ants: 5 < t_tired < 10, 0.2 < c < 0.21
# middle aged ants: 20 < t_tired < 30, 0.1 < c < 0.15
# young ants: 40 < t_tired < 50, 0.05 < c < 0.1
# I want to plot probability of finding food for each group of ages and number of ants eaten

with open('old_ants.data') as f_old:
    lines = f_old.readlines()
    x_old = [int(line.split()[0]) for line in lines]
    old_eaten = [int(line.split()[2]) for line in lines]

with open('middle_ants.data') as f_middle:
    lines = f_middle.readlines()
    x_middle = [int(line.split()[0]) for line in lines]
    middle_eaten = [int(line.split()[2]) for line in lines]

with open('young_ants.data') as f_young:
    lines = f_young.readlines()
    x_young = [int(line.split()[0]) for line in lines]
    young_eaten = [int(line.split()[2]) for line in lines]

n_sim = len(x_old)
p_old = x_old.count(1)/n_sim
p_middle = x_middle.count(1)/n_sim
p_young = x_young.count(1)/n_sim

fig1, ax1 = plt.subplots(1)
fig1.suptitle('probability of finding food')
print("p_old = " + str(p_old))
print("p_middle = " + str(p_middle))
print("p_young = " + str(p_young))
width = 0.1
ax1.bar("old ants", p_old, width, color="green")
ax1.bar("middle aged \n ants", p_middle, width, color="red")
ax1.bar("young ants", p_young, width, color="blue")
margin = (3/2 - width) + width / 3
ax1.set_xlim(-margin, 3 - 1 + margin)
ax1.set_ylim(0, 1.1)
plt.ylabel("p")
plt.show()


