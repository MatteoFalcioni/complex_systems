import matplotlib.pyplot as plt

to_plot = open("sim.data", "r")
for row in to_plot:
    row = row.split(' ')
    names.append(row[0])
    marks.append(int(row[1]))








