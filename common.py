import pandas as pd
import numpy as np

def read_input(filename):
    with open(filename) as f:
        cities = []
        for line in f.readlines()[1:]:  # Ignore the first line.
            xy = line.split(',')
            cities.append([float(xy[0]), float(xy[1])])

    with open(filename) as f:   
        cities_array = np.loadtxt(filename, delimiter=',', dtype='float',skiprows=1,usecols=[0,1])

    cities_df = pd.read_csv(filename)


    return cities, cities_array, cities_df


def format_solution(solution):
    return 'index\n' + '\n'.join(map(str, solution))


def print_solution(solution):
    print(format_solution(solution))


def search_center(cities):
    '''
    都市の座標の中央点を返す関数

    x_mean:
        中央点のx座標
    x_mean:
        中央点のy座標
    '''
    x_total = 0
    y_total = 0
    for city in cities:
        x_total = x_total + city[0]
        y_total = y_total + city[1]
    x_mean = x_total / len(cities)
    y_mean = y_total / len(cities)
    return (x_mean, y_mean)

class KMeans(object):
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)