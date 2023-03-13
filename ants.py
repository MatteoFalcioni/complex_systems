import random
import math


# ants position. In the following called z=(x,y)
class Pos:

    def __init__(self, x, y):
        self.x = x
        self.y = y


psi = 5  # search range, could be a vector but in this model we take it as a number
phi = 7.5 / psi  # search diameter
p_nest = Pos(0.4, 0.4)  # nest position
p_food = Pos(0.7, 0.5)  # food position (coincides with global minimum of landscape potential f)
a = 15
b = 0.3  # a & b constants to fix weights of the exponential in model
w = 0.11  # frequency of nest->food->nest travel
r = 0.3  # chaotic annealing parameter: defines how quickly the colony synchronizes


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


