import os
import sys
import random
import numpy as np
from numpy.random import seed
seed(1)

if len(sys.argv) != 5:
    print('usage : ', sys.argv[0], 'data_file k(num of clusters) r(num of iterations) output_file')
    sys.exit()

X = np.loadtxt(sys.argv[1], delimiter=",")
Y = X.tolist()
print("\n")


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


def find_centers(x, k, r):
    quant_err = 100000
    final_res = []
    for i in range(r):
        # Initialize to K random centers
        oldmu = random.sample(x, k)
        # print("oldmu", oldmu)
        mu = random.sample(x, k)
        # print(mu)
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