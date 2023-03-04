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
b = 0.5  # a & b constants to fix weights of the exponential in model
w = 0.11  # frequency of nest->food->nest travel
D = 0.3  # distance below which ants are connected in graph. Could also grow with t (alarm pheromone density increases)
M = 5  # number of recruited ants
N = 50  # number of searching ants (whole colony: N+M)
K = N+M
t_max = 500  # number of iterations
colony = []  # whole colony
s_0 = 0.999  # initial value of organization parameter
min_t_h = 5  # minimum value for homing times
max_t_h = 20  # maximum value for homing times
min_c = 0.1  # minimum value for nest constant
max_c = 0.3  # maximum value for nest constant
predator = False  # presence of predator in the environment
food_found = False  # to break out of food searching loop


class Ant:

    def __init__(self, s, z, t_h, c, v):
        self.s = s  # colony synchronization parameter
        self.z = z  # position (2D)
        self.t_h = t_h  # homing time
        self.c = c  # constant for nest position
        self.homing = False  # boolean value to check if the ant is homing
        self.v = v  # chaotic search center
        self.alarm = False  # when True, it means that the ant is alarmed by a predator attack

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

    def optimal_path(self, t):  # when y=0 model approximates to these equations
        self.z.x = self.z.x + math.exp(b) * (math.sin(w*t) * (p_food.x - p_nest.x) - (self.z.x - p_nest.x))
        self.z.y = self.z.y + math.exp(b) * (math.sin(w * t) * (p_food.y - p_nest.y) - (self.z.y - p_nest.y))


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
    if rng <= phi / 2:
        pass
    else:
        print("You are generating a position farther than the search radius from the nest, which is useless")
    x_n = center.x
    y_n = center.y
    x_gen = random.uniform(x_n - rng, x_n + rng)
    y_gen = random.uniform(y_n - rng, y_n + rng)
    z_gen = Pos(x_gen, y_gen)
    return z_gen


# plots
plotX1 = [[0 for j in range(t_max)] for i in range(N+M)]
plotY1 = [[0 for q in range(t_max)] for p in range(N+M)]
plotX2 = [[0 for m in range(t_max)] for n in range(N+M)]
plotY2 = [[0 for b in range(t_max)] for a in range(N+M)]
plotX3 = [[0 for m in range(t_max)] for n in range(N+M)]
plotY3 = [[0 for b in range(t_max)] for a in range(N+M)]
plotX4 = [[0 for m in range(t_max)] for n in range(N+M)]
plotY4 = [[0 for b in range(t_max)] for a in range(N+M)]
plotX_h = [[0 for m in range(t_max)] for n in range(N+M)]
plotY_h = [[0 for b in range(t_max)] for a in range(N+M)]
plotGraphX = [0] * (N+M+1)
plotGraphY = [0] * (N+M+1)

for i in range(N+M):  # filling colony
    T_h = random.randint(min_t_h, max_t_h)  # generate t_h
    C = random.uniform(min_c, max_c)  # generate c
    ant_gen = Ant(s_0, pos_generator(0.1, p_nest), T_h, C, v_search)  # generate ants around the nest
    colony.append(ant_gen)
colony.append(Ant(0, p_nest, 0, 0, 0))
# "source" ant, will be fixed in the nest to represent nest position in following graph

A = [[0 for j in range(N+M+1)] for i in range(N+M+1)]
# Adjacency matrix. the +1 is for the "source" of the alarm, which is an ant fixed in the nest
predator_pos = pos_generator(phi/2, p_nest)  # positioning predator randomly in the square where ants are wandering
if predator:
    colony[K+1].alarm = True  # when predator appears, the ants in the nest are alarmed (add time delay for this info)

T = [0] * N  # list of counters for homing times
TH = [0] * N  # counter to check how much time it takes for ants to reach their nest

for t1 in range(t_max):  # food search loop (+ homing)
    if not predator:
        r = random.uniform(0, 1)
        if r < 0.1:
            predator = True
            print("predator spawned in (" + str(predator_pos.x) + ", " + str(predator_pos.y) + ")")

    print("t1= " + str(t1))
    # only N ants moving now, M will join when food is found
    for i in range(N):  # first, checking which ants have reached their homing time
        # could be that ants can't find their nest in time for T to be again equal to t_h, so needed to check
        if colony[i].t_h == T[i] and not colony[i].homing:
            T[i] = 0
            TH[i] = 0
            colony[i].homing = True  # ant i is now heading home instead of looking for food
            # centering chaotic search around current position to look for the nest in the homing process
            v_home = Pos(7.5 / (2 * psi) - colony[i].z.x, 7.5 / (2 * psi) - colony[i].z.y)
            colony[i].v = pos_generator(0.1, v_home)  # v_home.x & v_home.y cannot be equal or z(t) will be a line

    for i in range(N):
        if not colony[i].homing:  # if ant i is not heading home, then it should be looking for food
            T[i] += 1  # ants get tired when searching for food (not while homing)
            colony[i].model(t1)  # no decrement of s here, r=0
            plotX1[i][t1] = colony[i].z.x
            plotY1[i][t1] = colony[i].z.y
            if landscape(colony[i].z) < delta:
                print("*** ant " + str(i) + " has found food in z = (" + str(colony[i].z.x) + ", " +
                      str(colony[i].z.y) + ") after " + str(t1) + "time steps  ***")
                food_found = True
                break
        else:  # if homing
            colony[i].model(t1)  # no decrement of s here, r=0
            TH[i] += 1
            plotX_h[i][t1] = colony[i].z.x
            plotY_h[i][t1] = colony[i].z.y
            if dist(colony[i].z, p_nest) < colony[i].c:
                # print("ant " + str(i) + " has found nest in z = (" + str(colony[i].z.x) + ", " +
                #      str(colony[i].z.y) + ") after " + str(TH[i]) + " time steps. It's nest constant was " +
                #      str(colony[i].c))
                colony[i].homing = False
                colony[i].v = v_search
    for i in range(N + M + 1):  # adjourning adjacency matrix at each time step
        for j in range(N + M + 1):
            if j < i:  # undirected graph -> symmetric adjacency matrix -> i<j
                if dist(colony[i].z, colony[j].z) <= D:
                    A[i][j] = 1
                    A[j][i] = 1  # symmetric
                else:
                    A[i][j] = 0
                    A[j][i] = 0
        if t1 == 499:
            plotGraphX[i] = colony[i].z.x
            plotGraphY[i] = colony[i].z.y
    if predator:
        for i in range(K+1):  # check if ant K = nest
            if A[K][i] == 1:  # ants connected with nest start getting predator alarm
                # ants should notice predator when other ants disappear: fix this
                colony[i].alarm = True  # check if saying colony[i] is the same as saying A[i][j] for all j
        for i in range(K+1):
            for j in range(K+1):
                if colony[i].alarm and A[i][j] == 1:
                    colony[j].alarm = True

    # if food_found:
    #    break

# ant that found food goes back and recruits M more ants to follow on the search
t_graph1 = 20
t_graph2 = 200
if food_found:
    for t2 in range(t_max):
        for i in range(N+M):
            colony[i].chaotic_annealing()  # decrement of s
            colony[i].model(t2)
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
for i in range(N+M):
    plt.scatter(plotX2[i], plotY2[i], marker=".", s=30)
plt.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
plt.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('z(t)')
plt.show()
for i in range(N+M):
    plt.scatter(plotX3[i], plotY3[i], marker=".", s=30)
plt.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
plt.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('z(t)')
plt.show()
for i in range(N+M):
    plt.scatter(plotX4[i], plotY4[i], marker=".", s=30)
plt.scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
plt.scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('z(t)')
plt.show()
"""
# plots
"""
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
for i in range(N+M+1):
    if colony[i].alarm:
        plt.scatter(plotGraphX[i], plotGraphY[i], marker=".", s=85, color="red")
    else:
        plt.scatter(plotGraphX[i], plotGraphY[i], marker=".", s=85, color="green")
    for j in range(N+M+1):
        if j < i and A[i][j] == 1:
            x_values = [plotGraphX[i], plotGraphX[j]]
            y_values = [plotGraphY[i], plotGraphY[j]]
            plt.plot(x_values, y_values, color="lightskyblue")
plt.scatter(predator_pos.x, predator_pos.y, marker="*", s=80, color="black")
plt.xlabel('x')
plt.ylabel('y')
plt.title('ant network at t=499')
plt.show()

"""
fig1, axs1 = plt.subplots(1, 2)
fig1.suptitle('z(t) of food searching ants (left) and homing ones (right)')
for i in range(N+M):
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

for i in range(N+M):
    axs[0].scatter(plotX2[i], plotY2[i], marker=".", s=30)
    axs[1].scatter(plotX3[i], plotY3[i], marker=".", s=30)
    axs[2].scatter(plotX4[i], plotY4[i], marker=".", s=30)
axs[0].scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
axs[1].scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
axs[2].scatter(p_nest.x, p_nest.y, marker="*", s=40, color="black")
axs[0].scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
axs[1].scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
axs[2].scatter(p_food.x, p_food.y, marker="*", s=40, color="black")
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








