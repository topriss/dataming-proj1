#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from __future__ import print_function, division, absolute_import

from codecs import open as open

import numpy as np
from sklearn import manifold
from matplotlib import pyplot as plt
import pandas as pd

from cluster import *
from plot import *


def main():
  data_fname = 'data/data_iris_flower/iris.data'
  cls_rst_fname = 'tmp.txt'

  # Load data. Missing values should be handled beforehand
  df_pre = []
  for data_line in open(data_fname, 'r', 'utf-8').readlines():
    data_sample = list(map(float, data_line.strip().split()))
    df_pre.append(data_sample)

  num_feature = len(df_pre[0])
  df = {}
  for i in range(1, 1 + num_feature):
    df['feature_{}'.format(i)] = []
  for data_sample in df_pre:
    for i in range(num_feature):
      df['feature_{}'.format(i + 1)].append(data_sample[i])

  # Load cluster result
  df['cls_rst'] = []
  for line in open(cls_rst_fname, 'r', 'utf-8').readlines():
    df['cls_rst'].append(line.strip())

  # plotting
  df = pd.DataFrame(df)
  rad_viz = pd.plotting.radviz(df, 'cls_rst')
  figure_fig = plt.gcf()  # 'get current figure'
  figure_fig.savefig('figure.eps', format='eps', dpi=1000)

  plt.show()

  return


if __name__ == '__main__':
  NUM_CORE = 4

  main()
