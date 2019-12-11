import os
import sys
import random
import numpy as np
from numpy import newaxis, delete
from numpy.random import seed
seed(1)

if len(sys.argv) != 5:
    print('usage : ', sys.argv[0], 'data_file k(num of clusters) r(num of iterations) output_file')
    sys.exit()

X = np.loadtxt(sys.argv[1], delimiter=",")
Y = X.tolist()
print("\n")
mu = []
oldmu = []

def cluster_points(X, mu):
    X = np.asarray(X)
    mu = np.asarray(mu)
    clusters = {}
    res = []
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) \
                         for i in enumerate(mu)], key=lambda t: t[1])[0]
        #print(bestmukey)
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
        res.append(bestmukey)
    return clusters, res

def findQ_err(X, mu, res):
    X = np.asarray(X)
    err = 0
    count = 0
    for x in X:
        err += np.linalg.norm(x - mu[res[count]])
        count += 1
    return err


def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis=0))
    return newmu


def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])

def find_init_centers(x, k):
    x = np.asarray(x)
    centroids = x.copy()
    np.random.shuffle(centroids)
    unselected_points = centroids[1:].copy()
    distances = np.zeros(len(unselected_points))
    for i in range(1, k):
        distances += ((unselected_points - centroids[i]) ** 2).sum(axis=1)
        furtherest_point_index = np.argmax(distances)
        centroids[i] = unselected_points[furtherest_point_index]
        # delete selected points from distances and unselected_points
        distances = delete(distances, furtherest_point_index)
        unselected_points = delete(unselected_points, furtherest_point_index, 0)
    return centroids[:k]


def find_centers(x, k, r):
    # Initialize to K random centers
    global oldmu, mu
    quant_err = 100000
    final_res = []
    for i in range(r):
        oldmu = random.sample(x, k)
        mu = find_init_centers(x, k)
        while not has_converged(mu, oldmu):
            oldmu = mu
            # Assign all points in X to clusters
            clusters, res = cluster_points(x, mu)
            # Reevaluate centers
            mu = reevaluate_centers(oldmu, clusters)

        newQuant_err = findQ_err(x, mu, res)
        if newQuant_err < quant_err:
             quant_err = newQuant_err
             final_res = res
    return quant_err, final_res


Q_err, res = find_centers(Y, int(sys.argv[2]), int(sys.argv[3]))
res = np.asarray(res)
print(res)
print(Q_err)

with open(sys.argv[4], 'w') as f:
    for item in res:
        f.write("%s\n" % item)