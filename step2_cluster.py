#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import logging
from plot import *
from cluster import *


def plot(data, density_threshold, distance_threshold, auto_select_dc=False):
  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
  dpcluster = DensityPeakCluster()
  rho, delta, nneigh = dpcluster.cluster_func(
    load_paperdata, data, density_threshold, distance_threshold, auto_select_dc=auto_select_dc)
  logger.info(str(len(dpcluster.ccenter)) + ' center as below')
  for center in dpcluster.ccenter:
    logger.info('id={}, rho={}, delta={}'.format(center, rho[center], delta[center]))

  # save cluster result
  with open(r'./tmp.txt', 'w') as fo:
    fo.write('\n'.join(map(str, dpcluster.cluster[1:])))

  # plot_rho_delta(rho, delta)   #plot to choose the threthold
  # plot_cluster(dpcluster)

  return


if __name__ == '__main__':
  # plot('./data/data_in_paper/example_distances.dat', 20, 0.1)
  plot('./data/data_iris_flower/iris.forcluster',
       density_threshold=0.09,
       distance_threshold=0.005,
       auto_select_dc=True)
