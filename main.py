import random
import numpy as np
import math
import matplotlib.pyplot as plt


# ants position. In the following called z=(x,y)
class Pos:

    def __init__(self, x, y):
        self.x = x
        self.y = y


psi = 5  # search range, could be a vector but in this model we take it as a number
phi = 7.5 / psi  # search diameter
p_nest = Pos(0.4, 0.4)  # nest position
p_food = Pos(0.7, 0.5)  # food position (coincides with global minimum of landscape potential f)
v_search = Pos(7.5 / (2 * psi) - p_nest.x, 7.5 / (2 * psi) - p_nest.y)  # needed in the model to roam both z>0 & z<0
r = 0.3  # chaotic annealing parameter: defines how quickly the colony synchronizes
delta = 1e-3  # upper limit on landscape potential (f(z)<delta -> z minimum)
a = 15
b = 0.3  # a & b constants to fix weights of the exponential in model
w = 0.11  # frequency of nest->food->nest travel
D = 0.2  # distance below which ants are connected in graph. Could also grow with t (alarm pheromone density increases)
M = 5  # number of recruited ants
N = 50  # number of searching ants (whole colony: K)
K = N+M
t_max = 1000  # number of iterations
s_0 = 0.999  # initial value of organization parameter
min_t_h = 40  # minimum value for homing times
max_t_h = 50  # maximum value for homing times
min_c = 0.05  # minimum value for nest constant
max_c = 0.1  # maximum value for nest constant
pred_prob = 0.1  # probability of predator appearing at each time step
predator_rng = 0.2  # radius of the circle in which predator eats ants
pred_time = 50  # time steps for which the predator will be present in the environment
time_delay = 5  # to give delay to info of ants being eaten by predator
sim_number = 1000  # number of simulations to perform

optimal_path_finding = False  # put it false if you don't want the ants to look for optimal path, and just look for food
predation_effect = True  # put it false if you don't want predator to ever appear during food searching


class Ant:

    def __init__(self, s, z, t_h, c, v):
        self.s = s  # colony synchronization parameter
        self.z = z  # position (2D)
        self.t_h = t_h  # homing time
        self.c = c  # constant for nest position
        self.v = v  # chaotic search center
        self.homing = False  # boolean value to check if the ant is homing
        self.alarm = False  # when True, it means that the ant is alarmed by a predator attack
        self.alive = True  # to check if ant i has been eaten by predator or not
        self.safe = False  # to check if ant i has returned to the nest safely

    def chaotic_map(self):  # map to try out chaotic crawling. needed v to roam both z>0 & z<0
        self.z.x = self.z.x * math.exp(3 - psi * self.z.x)
        self.z.y = self.z.y * math.exp(3 - psi * self.z.y)

    def chaotic_annealing(self):  # variable s gradually decreases -> colony gradually organizes
        self.s = math.pow(self.s, (1 + r))

    def chaotic_crawling(self):  # when initially searching for food, r=0 and the model approximates to these eq.
        self.z.x = (self.z.x + self.v.x) * math.exp(3 - psi * (self.z.x + self.v.x)) - self.v.x
        self.z.y = (self.z.y + self.v.y) * math.exp(3 - psi * (self.z.y + self.v.y)) - self.v.y

    def model(self, t):
        self.z.x = (self.z.x + self.v.x) * math.exp((1 - math.exp(-a * self.s)) * (3 - psi * (self.z.x + self.v.x))) + \
            (abs(math.sin(w * t)) * (p_food.x - p_nest.x) - (self.z.x - p_nest.x)) * \
            math.exp((-2 * a * self.s) + b) - self.v.x
        self.z.y = (self.z.y + self.v.y) * math.exp((1 - math.exp(-a * self.s)) * (3 - psi * (self.z.y + self.v.y))) + \
            (abs(math.sin(w * t)) * (p_food.y - p_nest.y) - (self.z.y - p_nest.y)) * \
            math.exp((-2 * a * self.s) + b) - self.v.y
# t current time step
# iff a is very big (circa > 12) chaotic_crawling() is equivalent to model().


def dist(pos_1, pos_2):
    x1 = pos_1.x
    x2 = pos_2.x
    y1 = pos_1.y
    y2 = pos_2.y

    return math.sqrt(math.pow((x1 - x2), 2) + math.pow((y1 - y2), 2))


# landscape potential (energy function to minimize during search)
def landscape(pos):
    x = pos.x
    y = pos.y
    f: float = (math.pow((x - 0.7), 2) * (0.1 + math.pow((0.6 + y), 2))) \
        + (math.pow(y - 0.5, 2) * (0.15 + math.pow((0.4 + x), 2)))
    return f


def pos_generator(rng, center):
    # generate a random position on a square of L=rng around the nest
    # useless to generate it farther than phi/2 from p_nest since psi forces the ant to stay in the square of L=phi/2
    x_n = center.x
    y_n = center.y
    x_gen = random.uniform(x_n - rng, x_n + rng)
    y_gen = random.uniform(y_n - rng, y_n + rng)
    z_gen = Pos(x_gen, y_gen)
    return z_gen

"""
# plots
plotX1 = [[0 for j in range(t_max)] for i in range(K)]
plotY1 = [[0 for j in range(t_max)] for i in range(K)]
plotX2 = [[0 for j in range(t_max)] for i in range(K)]
plotY2 = [[0 for j in range(t_max)] for i in range(K)]
plotX3 = [[0 for j in range(t_max)] for i in range(K)]
plotY3 = [[0 for j in range(t_max)] for i in range(K)]
plotX4 = [[0 for j in range(t_max)] for i in range(K)]
plotY4 = [[0 for j in range(t_max)] for i in range(K)]
plotX_h = [[0 for j in range(t_max)] for i in range(K)]
plotY_h = [[0 for j in range(t_max)] for i in range(K)]
plotGraphX1 = [0] * (K+1)
plotGraphY1 = [0] * (K+1)
plotGraphX2 = [0] * (K+1)
plotGraphY2 = [0] * (K+1)
plotGraphXF = [0] * (K+1)
plotGraphYF = [0] * (K+1)
"""

t_graph1 = 20
t_graph2 = 50

if predation_effect:
    simulation_data = open('young_ants.data', 'a')
else:
    simulation_data = open('sim.data', 'a')
print("starting to perform " + str(sim_number) + " simulations \n")

for sim in range(sim_number):
    food_found = False  # to break out of food searching loop
    food_found_time = 0
    predator = False  # presence of predator in the environment. Starting w\out predator
    colony = []  # whole colony

    for i in range(K):  # filling colony
        T_h = random.randint(min_t_h, max_t_h)  # generate t_h
        C = random.uniform(min_c, max_c)  # generate c
        ant_gen = Ant(s_0, pos_generator(0.05, p_nest), T_h, C, v_search)  # generate ants around the nest
        colony.append(ant_gen)
    colony.append(Ant(0, p_nest, 0, 0, 0))
    # "source" ant, will be fixed in the nest to represent nest position in following graph
    # so K+1-th ant is the source of the alarm in the following

    A = [[0 for j in range(K+1)] for i in range(K+1)]  # Adjacency matrix
    # positioning predator randomly in the square where ants are wandering, but not too near to the nest
    predator_pos = p_nest
    while p_nest.x - max_c < predator_pos.x < p_nest.x + max_c or p_nest.y - max_c < predator_pos.y < p_nest.y + max_c:
        predator_pos = pos_generator(phi/2, Pos(p_nest.x, p_nest.y))

    predation_time_interval = [-1, -1]  # to check when predator has arrived and when it has left
    pred_staying = 0  # counter to check in real time if predator is there
    time_wait = 0
    safe_ants = 0
    ants_eaten = 0

    T = [0] * N  # counter to check if homing times have been reached by ants
    TH = [0] * N  # counter to check how much time it takes for ants to reach their nest when homing

    for t1 in range(t_max):  # food searching loop (+ homing + predation)
        # print("t1= " + str(t1))
        food_found_time += 1

        # adjourning ants network. consider K-th element which is the source
        for i in range(K + 1):  # adjourning adjacency matrix at each time step
            for j in range(K + 1):
                if j < i:  # undirected graph -> symmetric adjacency matrix -> j<i is enough
                    if dist(colony[i].z, colony[j].z) <= D:
                        A[i][j] = 1
                        A[j][i] = 1  # symmetric
                    else:
                        A[i][j] = 0
                        A[j][i] = 0
            # if t1 == t_max-1:
            #     plotGraphXF[i] = colony[i].z.x
            #     plotGraphYF[i] = colony[i].z.y

        # spawning predator with probability pred_prob at each step
        if predation_effect:
            if not predator:
                # when pred_staying > pred_time it means that predator has spawned and has left after pred_time
                if pred_staying < pred_time:
                    r = random.uniform(0, 1)
                    if r < pred_prob:
                        predator = True
                        # print("predator spawned in (" + str(predator_pos.x) + ", " + str(predator_pos.y) + ") at t=" + str(t1))

                else:
                    for i in range(K+1):
                        colony[i].safe = False
                        colony[i].alarm = False
            else:
                # if predator already present
                pred_staying += 1
                for i in range(K):
                    if dist(colony[i].z, predator_pos) <= predator_rng and colony[i].alive:
                        colony[i].alive = False  # predator eats ants which come closer than predator_rng to him
                        # print("ant " + str(i) + " has been eaten")
                        ants_eaten += 1
                if ants_eaten > 0:
                    time_wait += 1
                if time_wait >= time_delay:
                    if not colony[K].alarm:
                        colony[K].alarm = True  # when predator spawns, ants in the nest are alarmed (w/ a time delay)
                    for i in range(K+1):
                        if A[K][i] == 1 and not colony[i].alarm:  # ants connected with nest start getting predator alarm
                            colony[i].alarm = True
                            colony[i].homing = True
                for i in range(K+1):
                    for j in range(K+1):
                        if colony[i].alarm and A[i][j] == 1:  # alarm propagates around the network
                            colony[j].alarm = True
                            colony[j].homing = True
                # notice that information about predator being in the environment propagates instantaneously around the
                # network. It would be more realistic if it had a certain time of propagation, or maybe if it was a continuously
                # increasing function
                if pred_staying == pred_time:
                    predator = False
                    predation_time_interval = [t1 - pred_staying, t1]
                    # print("**** predator disappeared from environment after " + str(pred_staying) + " time steps ****")
                    # print("since predator has disappeared, all alive ants are now going to look for food again")
                    # print("predator time interval was [" + str(predation_time_interval[0])
                    #      + ", " + str(predation_time_interval[1]) + "]")

        # only N ants moving now, M will join when food is found. Until then, M ants will stay in the nest
        for i in range(N):  # first, checking which ants have reached their homing time
            if colony[i].alive and not colony[i].safe:
                if not colony[i].alarm:  # if they are not running from predator
                    if not colony[i].homing:  # if they are not already in the homing process
                        if colony[i].t_h == T[i]:  # and they have reached their homing time
                            T[i] = 0  # reset counter of how tired ant i is (since ant i will now head home)
                            TH[i] = 0
                            colony[i].homing = True  # ant i is now heading home instead of looking for food
                            # centering chaotic search around current position to look for the nest in the homing process
                            v_home = Pos(7.5 / (2 * psi) - colony[i].z.x, 7.5 / (2 * psi) - colony[i].z.y)
                            # v_home.x & v_home.y cannot be equal or z(t) will be a line
                            colony[i].v = pos_generator(0.1, v_home)
                        else:  # if homing time has not been reached, look for food
                            T[i] += 1  # ants get tired only when searching for food (not while homing)
                            colony[i].model(t1)  # dynamics (no decrement of s here, r=0)
                            # plotX1[i][t1] = colony[i].z.x
                            # plotY1[i][t1] = colony[i].z.y
                            if landscape(colony[i].z) < delta:  # food has been found
                                # print("*** ant " + str(i) + " has found food in z = (" + str(colony[i].z.x) + ", " +
                                #       str(colony[i].z.y) + ") after " + str(t1) + "time steps  ***")
                                food_found = True
                                break
                    else:  # if homing
                        colony[i].model(t1)  # dynamics (no decrement of s here, r=0)
                        TH[i] += 1
                        # plotX_h[i][t1] = colony[i].z.x
                        # plotY_h[i][t1] = colony[i].z.y
                        if dist(colony[i].z, p_nest) < colony[i].c:
                            # print("ant " + str(i) + " has found nest in z = (" + str(colony[i].z.x) + ", " +
                            #       str(colony[i].z.y) + ") after " + str(TH[i]))
                            colony[i].homing = False
                            colony[i].v = v_search
                else:  # if ant i is alarmed by the presence of a predator it will be homing to escape
                    colony[i].homing = True
                    # centering chaotic search around current position to look for the nest in the homing process
                    v_home = Pos(7.5 / (2 * psi) - colony[i].z.x, 7.5 / (2 * psi) - colony[i].z.y)
                    if colony[i].homing:  # if it hasn't found the nest yet in its run from the predator
                        colony[i].model(t1)
                        if dist(colony[i].z, p_nest) < colony[i].c:  # nest found. Stay there since you're alarmed
                            colony[i].homing = False
                            colony[i].safe = True
                            safe_ants += 1
        if food_found:
            break

    # print("of " + str(N) + " total ants in the colony, " + str(ants_eaten) + " were eaten by predator while " +
    #       str(safe_ants) + " returned home safely. " + str(abs(N - (ants_eaten + safe_ants))) +
    #       " ants are still out there...")
    ants_left = K - ants_eaten

    if not predation_effect:  # no predator: just save if food found & how much time it took
        if food_found:
            simulation_data.write("1 \t" + str(food_found_time) + "\n")
        else:
            simulation_data.write("0 \t -1 \n")
    else:  # save als o how many ants are left after predation
        if food_found:
            simulation_data.write("1 \t" + str(food_found_time) + "\t" + str(ants_left) + "\n")
        else:
            simulation_data.write("0 \t -1 \t" + str(ants_left) + "\n")

    # ant that found food goes back and recruits M more ants to follow on the search. K ants involved now
    if food_found and optimal_path_finding:
        for t2 in range(t_max):
            for i in range(K):
                colony[i].chaotic_annealing()  # decrement of s
                colony[i].model(t2)
                """
                if colony[i].alive:
                    if t2 < t_graph1:
                        plotX2[i][t2] = colony[i].z.x
                        plotY2[i][t2] = colony[i].z.y
                    if t_graph1 < t2 < t_graph2:
                        plotX3[i][t2] = colony[i].z.x
                        plotY3[i][t2] = colony[i].z.y
                    if t2 > t_graph2:
                        plotX4[i][t2] = colony[i].z.x
                        plotY4[i][t2] = colony[i].z.y
                """
    print(str(sim) + "/" + str(sim_number) + " simulations performed")

simulation_data.close()

# To do list: comparing probability of being eaten by predator of different aged ants (by varying c and t_h, since
# older ants have bigger c and smaller t, while younger viceversa

"""
for i in range(K):
    plt.scatter(plotX2[i], plotY2[i], marker=".", s=30)
plt.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
plt.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('z(t)')
plt.show()
for i in range(K):
    plt.scatter(plotX3[i], plotY3[i], marker=".", s=30)
plt.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
plt.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('z(t)')
plt.show()
for i in range(K):
    plt.scatter(plotX4[i], plotY4[i], marker=".", s=30)
plt.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
plt.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('z(t)')
plt.show()

for i in range(N):
    plt.scatter(plotX1[i], plotY1[i], marker=".", s=30)
plt.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
plt.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('z(t)')
plt.show()
for i in range(N):
    plt.scatter(plotX_h[i], plotY_h[i], marker=".", s=30)
plt.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
plt.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('z(t) of homing ants')
plt.show()
"""

"""
# plots

fig, ax = plt.subplots(1)
fig.suptitle('predation response')
for i in range(K):
    if colony[i].alive:
        if colony[i].alarm:
            if not colony[i].safe:
                ax.scatter(plotGraphXF[i], plotGraphYF[i], marker=".", s=85, color="red")
            else:
                ax.scatter(plotGraphXF[i], plotGraphYF[i], marker=".", s=85, color="blue")
        else:
            ax.scatter(plotGraphXF[i], plotGraphYF[i], marker=".", s=85, color="green")
        for j in range(K + 1):
            if j < i and A[i][j] == 1 and colony[j].alive and not colony[i].safe and not colony[j].safe:
                x_values = [plotGraphXF[i], plotGraphXF[j]]
                y_values = [plotGraphYF[i], plotGraphYF[j]]
                ax.plot(x_values, y_values, color="lightskyblue")
    else:
        ax.scatter(plotGraphXF[i], plotGraphYF[i], marker="x", s=55, color="black")

ax.add_patch(plt.Circle((predator_pos.x, predator_pos.y), predator_rng, color='orange', fill=False))
ax.add_patch(plt.Circle((p_nest.x, p_nest.y), max_c, color='blue', fill=False))
ax.scatter(predator_pos.x, predator_pos.y, marker="X", s=80, color="orange")
ax.scatter(p_nest.x, p_nest.y, marker="*", s=80, color="blue")
ax.set(xlabel='x(t)', ylabel='y(t)')
plt.title('ant network at tf')
plt.show()


fig1, axs1 = plt.subplots(1, 2)
fig1.suptitle('z(t) of food searching ants (left) and homing ones (right)')
for i in range(K):
    axs1[0].scatter(plotX1[i], plotY1[i], marker=".", s=30)
    axs1[1].scatter(plotX_h[i], plotY_h[i], marker=".", s=30)
axs1[0].scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
axs1[1].scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
for ax in axs1.flat:
    ax.set(xlabel='x(t)', ylabel='y(t)')
    ax.set(adjustable='box', aspect='equal')
    ax.label_outer()
plt.show()


fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0.5)
axs = gs.subplots(sharex=True, sharey=True)

for i in range(K):
    axs[0].scatter(plotX2[i], plotY2[i], marker=".", s=30)
    axs[1].scatter(plotX3[i], plotY3[i], marker=".", s=30)
    axs[2].scatter(plotX4[i], plotY4[i], marker=".", s=30)

for ax in axs.flat:
    ax.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
    ax.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
    ax.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
    ax.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
    ax.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
    ax.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
fig.suptitle('z(t) of ants during optimal path searching')
axs[0].set_title('t < ' + str(t_graph1))
axs[1].set_title(str(t_graph1) + ' < t < ' + str(t_graph2))
axs[2].set_title('t > ' + str(t_graph2))

for ax in axs.flat:
    ax.set(xlabel='x(t)', ylabel='y(t)')
    # ax.set(adjustable='box', aspect='equal')
    ax.label_outer()
plt.show()
"""



