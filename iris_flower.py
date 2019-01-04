# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import f1_score

from DPC.DPC import DPC


def preprocess():
  feature = pd.read_csv('data/iris_flower/iris.data')
  label = pd.read_csv('data/iris_flower/iris.label')
  cos_dist = cosine_distances(feature)
  
  feature.to_csv('data/iris_flower/feature.CSV')
  label.to_csv('data/iris_flower/label.CSV')
  np.save('data/iris_flower/cos_dist.npy', cos_dist)


def main():
  feature = pd.read_csv('data/iris_flower/feature.CSV')
  label = pd.read_csv('data/iris_flower/label.CSV')
  cos_dist = np.load('data/iris_flower/cos_dist.npy')
  label_name = 'label'
  
  dpc = DPC(dist_array=cos_dist)
  dpc.cal_decision_map()
  
  plt.scatter(x=dpc.rho, y=dpc.delta)
  # plt.show()
  
  dpc.do_cluster(n_center=3)
  
  rst_label = pd.DataFrame({
    'Unnamed: 0': list(range(dpc.n)),
    'rst_label':  [label[label_name][idx] for idx in dpc.label]})
  for_vis = feature.merge(rst_label, how='outer', on='Unnamed: 0')
  for_vis = for_vis.drop(columns=['Unnamed: 0'])
  pd.plotting.radviz(for_vis, 'rst_label',
                     color=['red', 'yellow', 'cyan', 'purple'])
  # plt.show()
  plt.savefig('result/iris_flower.png', dpi=600)
  
  print('f1_macro: {}\nf1_micro: {}'.format(
    f1_score(label[label_name], rst_label['rst_label'], average='macro'),
    f1_score(label[label_name], rst_label['rst_label'], average='micro')
  ))
  
  return


if __name__ == '__main__':
  main()
