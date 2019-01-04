# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, manhattan_distances
from sklearn.metrics import f1_score

from DPC.DPC import DPC


def preprocess():
  df = pd.read_csv('data/student_knowledge/student_knowledge.CSV')
  df = df.loc[1:]  # headers
  df.index = pd.RangeIndex(start=0, stop=len(df), step=1)
  
  feature = df[df.columns[:-1]]
  label = pd.DataFrame(df[df.columns[-1]])
  cos_dist = cosine_distances(feature)
  euc_dist = euclidean_distances(feature)
  man_dist = manhattan_distances(feature)
  
  feature.to_csv('data/student_knowledge/feature.CSV')
  label.to_csv('data/student_knowledge/label.CSV')
  np.save('data/student_knowledge/cos_dist.npy', cos_dist)
  np.save('data/student_knowledge/euc_dist.npy', euc_dist)
  np.save('data/student_knowledge/man_dist.npy', man_dist)
  
  return


def main():
  # load
  feature = pd.read_csv('data/student_knowledge/feature.CSV')
  label = pd.read_csv('data/student_knowledge/label.CSV')
  dist = np.load('data/student_knowledge/euc_dist.npy')
  label_name = 'UNS'
  
  dpc = DPC(dist_array=dist)
  dpc.cal_decision_map()
  
  plt.scatter(x=dpc.rho, y=dpc.delta)
  # plt.show()
  
  dpc.do_cluster(n_center=5)
  
  rst_label = pd.DataFrame({
    'Unnamed: 0': list(range(dpc.n)),
    'rst_label':  [label[label_name][idx] for idx in dpc.label]})
  for_vis = feature.merge(rst_label, how='outer', on='Unnamed: 0')
  for_vis = for_vis.drop(columns=['Unnamed: 0'])
  pd.plotting.radviz(for_vis, 'rst_label',
                     color=['red', 'yellow', 'cyan', 'purple'])
  # plt.show()
  plt.savefig('result/stu_euc.png', dpi=600)
  
  print('f1_macro: {}\nf1_micro: {}'.format(
    f1_score(label[label_name], rst_label['rst_label'], average='macro'),
    f1_score(label[label_name], rst_label['rst_label'], average='micro')
  ))
  
  return


if __name__ == '__main__':
  # preprocess()
  main()
