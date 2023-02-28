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
p_nest = Pos(0.40, 0.40)  # nest position
p_food = Pos(0.7, 0.5)  # food position (coincides with global minimum of landscape potential f)
v_search = Pos(7.5 / (2 * psi) - p_nest.x, 7.5 / (2 * psi) - p_nest.y)  # needed in the model to roam both z>0 & z<0
r = 0.3  # chaotic annealing parameter: defines how quickly the colony synchronizes
delta = 0.5 * 1e-3  # upper limit on landscape potential (f(z)<delta -> z minimum)


class Ant:

    def __init__(self, s, z, t_h, c, v):
        self.s = s  # colony synchronization parameter
        self.z = z  # position (2D)
        self.t_h = t_h  # homing time
        self.c = c  # constant for nest position
        self.homing = False  # boolean value to check if the ant is homing
        self.v = v  # chaotic search center

    def chaotic_map(self):  # map to try out chaotic crawling. needed v to roam both z>0 & z<0
        self.z.x = self.z.x * math.exp(3 - psi * self.z.x)
        self.z.y = self.z.y * math.exp(3 - psi * self.z.y)

    def chaotic_annealing(self):  # variable s gradually decreases -> colony gradually organizes
        self.s = math.pow(self.s, (1 + r))

    def chaotic_crawling(self):  # when initially searching for food, r=0 and the model approximates to these eq.
        self.z.x = (self.z.x + self.v.x) * math.exp(3 - psi * (self.z.x + self.v.x)) - self.v.x
        self.z.y = (self.z.y + self.v.y) * math.exp(3 - psi * (self.z.y + self.v.y)) - self.v.y

    def model(self, a, b, w, t):
        self.z.x = (self.z.x + self.v.x) * math.exp((1 - math.exp(-a * self.s)) * (3 - psi * (self.z.x + self.v.x))) + \
            (abs(math.sin(w * t)) * (p_food.x - p_nest.x) - (self.z.x - p_nest.x)) * \
            math.exp((-2 * a * self.s) + b) - self.v.x
        self.z.y = (self.z.y + self.v.y) * math.exp((1 - math.exp(-a * self.s)) * (3 - psi * (self.z.y + self.v.y))) + \
            (abs(math.sin(w * t)) * (p_food.y - p_nest.y) - (self.z.y - p_nest.y)) * \
            math.exp((-2 * a * self.s) + b) - self.v.y
# w = frequency of nest->food->nest; a & b constants to fix weights of the exponential; t current time step


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
    if rng < phi / 2:
        pass
    else:
        print("You are generating a position farther than the search radius from the nest, which is useless")
    x_n = center.x
    y_n = center.y
    x_gen = random.uniform(x_n - rng, x_n + rng)
    y_gen = random.uniform(y_n - rng, y_n + rng)
    z_gen = Pos(x_gen, y_gen)
    return z_gen


M = 5  # number of recruited ants
N = 50  # number of searching ants (whole colony: N+M)
t_max = 500  # number of iterations
colony = []  # whole colony
s_0 = 0.999  # initial value of organization parameter
min_t_h = 5  # minimum value for homing times
max_t_h = 20  # maximum value for homing times
min_c = 0.3  # minimum value for nest constant
max_c = 0.4  # maximum value for nest constant
# plots
plotX1 = [[0 for j in range(t_max)] for i in range(N)]
plotY1 = [[0 for q in range(t_max)] for p in range(N)]
plotX2 = [[0 for m in range(t_max)] for n in range(N+M)]
plotY2 = [[0 for b in range(t_max)] for a in range(N+M)]
plotX3 = [[0 for m in range(t_max)] for n in range(N+M)]
plotY3 = [[0 for b in range(t_max)] for a in range(N+M)]
plotX4 = [[0 for m in range(t_max)] for n in range(N+M)]
plotY4 = [[0 for b in range(t_max)] for a in range(N+M)]
plotX_h = [[0 for m in range(t_max)] for n in range(N+M)]
plotY_h = [[0 for b in range(t_max)] for a in range(N+M)]

for i in range(N+M):  # filling colony
    T_h = random.randint(min_t_h, max_t_h)  # generate t_h
    C = random.uniform(min_c, max_c)  # generate c
    ant_gen = Ant(s_0, pos_generator(0.1, v_search), T_h, C, v_search)
    colony.append(ant_gen)

food_found = False
T = [0] * N  # list of counters for homing times
TH = [0] * N  # counter to check how much time it takes for ants to reach their nest

for t1 in range(t_max):  # food search loop (+ homing)
    print("t1= " + str(t1))
    for i in range(N):  # first, checking which ants have reached their homing time
        # could be that ants can't find their nest in time for T to be again equal to t_h, so needed to check
        if colony[i].t_h == T[i] and not colony[i].homing:
            T[i] = 0
            TH[i] = 0
            colony[i].homing = True  # ant i is now heading home instead of looking for food
            print("ant " + str(i) + " is heading home")
            # centering chaotic search around current position to look for the nest in the homing process
            v_home = Pos(7.5 / (2 * psi) - colony[i].z.x, 7.5 / (2 * psi) - colony[i].z.y)
            colony[i].v = pos_generator(0.1, v_home)

    for i in range(N):
        if not colony[i].homing:  # if ant i is not heading home, then it should be looking for food
            T[i] += 1  # ants get tired when searching for food (not while homing)
            colony[i].chaotic_crawling()
            plotX1[i][t1] = colony[i].z.x
            plotY1[i][t1] = colony[i].z.y
            if landscape(colony[i].z) < delta:
                print("*** ant " + str(i) + " has found food in z = (" + str(colony[i].z.x) + ", " +
                      str(colony[i].z.y) + ") after " + str(t1) + "time steps  ***")
                food_found = True
                break
        else:  # if homing
            colony[i].chaotic_crawling()
            TH[i] += 1
            plotX_h[i][t1] = colony[i].z.x
            plotY_h[i][t1] = colony[i].z.y
            if dist(colony[i].z, p_nest) < colony[i].c:
                print("ant " + str(i) + " has found nest in z = (" + str(colony[i].z.x) + ", " +
                      str(colony[i].z.y) + ") after " + str(TH[i]) + " time steps. It's nest constant was " +
                      str(colony[i].c))
                colony[i].homing = False
                colony[i].v = v_search
    if food_found:
        break

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

# ant that found food goes back and recruits M more ants to follow on the search
# Also, check that homing is right
for t2 in range(t_max):
    for i in range(N+M):
        colony[i].chaotic_annealing()  # decrement of s
        colony[i].model(2.5, 0.5, 4, t2)  # math error: out of scope with exp
        if t2 < 20:
            plotX2[i][t2] = colony[i].z.x
            plotY2[i][t2] = colony[i].z.y
        if 20 < t2 < 100:
            plotX3[i][t2] = colony[i].z.x
            plotY3[i][t2] = colony[i].z.y
        if t2 > 300:
            plotX4[i][t2] = colony[i].z.x
            plotY4[i][t2] = colony[i].z.y


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




