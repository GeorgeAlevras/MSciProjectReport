import numpy as np

def get_real_data(population):
    with open('data.txt', 'r') as file:
            lines = file.readlines()
    prevelance = [float(lines[i].rstrip('\n')) for i in range(len(lines))]

    infected = population*0.01*np.array(prevelance)
    return list(infected)