#!/usr/bin/env python3

import sys
import math
import pprint

from common import print_solution, read_input, search_center

# データ加工・処理・分析ライブラリ
import numpy as np
import numpy.random as random
import scipy as sp
from pandas import Series, DataFrame
import pandas as pd
import math

# 可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# 機械学習ライブラリ
import sklearn

# k-means法を使うためのインポート
from sklearn.cluster import KMeans


# 都市の分類を行う（k-means++）
# k-means法を使うためのインポート
from sklearn.cluster import KMeans


def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def solve(cities, cities_array, cities_df):
    N = len(cities)

    # 巡回対象となる都市の数
    cities_num = cities_array.shape[0]

    # クラスターの数の場合分け
    if cities_num == 5 :
        cluster_num = 2
    elif cities_num == 8 or 16 :
        cluster_num = 4
    elif cities_num == 64 or 128 :
        cluster_num = 6
    elif cities_num == 512 :
        cluster_num = 7
    else:
        cluster_num = 8

    # KMeansクラスの初期化
    kmeans = KMeans(init='k-means++', n_clusters = cluster_num)

    # クラスターの重心を計算
    kmeans.fit(cities_array)

    # クラスター番号を予測
    y_pred = kmeans.predict(cities_array)

    # クラスター番号をリストcitiesに追加する
    cluster = y_pred.tolist()
    for i in range(N):
        cities[i].append(cluster[i])

    dist = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(N):
            dist[i][j] = dist[j][i] = distance(cities[i], cities[j])

    # 都市間の中央から最も近い都市を探す
    # 中央点の座標を計算
    center_position = search_center(cities)
    # 中央点と各種都市の距離を計算してリストに格納する
    dist_between_city_center = []
    for city in cities:
        diff = math.sqrt((city[0] - center_position[0])** 2 + (city[1] - center_position[1])** 2)
        dist_between_city_center.append(diff)
    # 中央点との距離が最も近い値のインデックス 
    current_city = dist_between_city_center.index(min(dist_between_city_center))

    # 始点のクラスタIDを取る
    current_cluster = cities[current_city][2]

    # 始点となった都市以外のインデックスのセットを作成する
    unvisited_cities = set(range(N))
    unvisited_cities.remove(current_city)

    # 始点となった都市のクラスタ以外のセットを作成する
    unvisited_city_clusters = set(cluster)
    unvisited_city_clusters.remove(current_cluster)

    solution = [current_city]

    current_cluster_cities = []
    for city in cities:
        if city[2] == current_cluster:
            current_cluster_cities.append(cities.index(city))
    # 同一クラスタ内で始点となった都市をリストから除去する
    current_cluster_cities.remove(current_city)


    def distance_from_current_city(to):
        return dist[current_city][to]

    # while len(unvisited_cities) != 0:
    while unvisited_cities:

        while current_cluster_cities:
            next_city = min(current_cluster_cities, key=distance_from_current_city)
            current_cluster_cities.remove(next_city)
            unvisited_cities.remove(next_city)
            solution.append(next_city)
            current_city = next_city
        
        if (unvisited_cities):
            next_city = min(unvisited_cities, key=distance_from_current_city)
            solution.append(next_city)
            current_cluster = cities[next_city][2]
            current_cluster_cities = []
            for city in cities:
                if city[2] == current_cluster:
                    current_cluster_cities.append(cities.index(city))
            current_cluster_cities.remove(next_city)
            unvisited_cities.remove(next_city)
            current_city = next_city
        
        else:
            break
        

    return solution

if __name__ == '__main__':
    assert len(sys.argv) > 1
    cities_tuple = read_input(sys.argv[1])
    solution = solve(cities_tuple[0], cities_tuple[1], cities_tuple[2])
    num = sys.argv[1].split('_')[1][0]
    output_csv = 'solution_yours_' + num + '.csv'
    with open(output_csv, 'w') as f:
        f.write('index'+'\n')
        for line in solution:
            f.write(str(line)+'\n')

