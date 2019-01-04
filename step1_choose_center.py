#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
from DPC_ori.plot import *
from DPC_ori.cluster import *
from matplotlib import pyplot as plt


def plot(data, auto_select_dc=False):
  logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  dpcluster = DensityPeakCluster()
  distances, max_dis, min_dis, max_id, rho = dpcluster.local_density(
    load_paperdata, data, auto_select_dc=auto_select_dc)
  delta, nneigh = min_distance(max_id, max_dis, distances, rho)
  plot_rho_delta(rho, delta)  # plot to choose the threthold

  # trying to auto select cluster center
  gamma = [list(z) for z in zip(delta * rho, rho, range(len(rho)))]
  gamma = sorted(gamma, key=lambda x: x[1])
  gamma = sorted(gamma, key=lambda x: x[0])
  plt.hist([g[0] for g in gamma])
  plt.show()

  # plt.hist(rho, bins=list(range(-1, 9)))
  # plt.show()

  return


if __name__ == '__main__':
  # plot('./data/data_in_paper/example_distances.dat')
  plot('./data/data_iris_flower/iris.forcluster', auto_select_dc=True)
  # plot('./data/data_iris_flower/iris.eucdist', auto_select_dc=True)
