# -*- coding: UTF-8 -*-
import numpy as np


class DPC(object):

  def __init__(self, dist_array):
    """
    :param dist_array: pre-calculated pairwise distance array
    """
    super(DPC, self).__init__()
    assert len(dist_array.shape) == 2
    assert dist_array.shape[0] == dist_array.shape[1]
    self.dist_array = dist_array.astype(float)
    self.max_dist = dist_array.max()
    self.min_dist = dist_array.min()
    self.n = dist_array.shape[0]

    self.rho = None
    self.delta = None
    self.father = None
    self.label = None

  def cal_decision_map(self):
    # step 1: cal cut distance
    max_dist, min_dist = self.max_dist, self.min_dist
    cut_dist = (max_dist + min_dist) / 2
    delta_dist = max_dist - min_dist
    while True:
      p = (self.dist_array < cut_dist).sum() / float(self.n ** 2)
      # criteria according to paper
      if 0.01 <= p <= 0.02:
        break
      # bin search
      if p < 0.01:
        min_dist = cut_dist
      else:
        max_dist = cut_dist
      cut_dist = (max_dist + min_dist) / 2
      # handle extreme cases
      if max_dist - min_dist < 1e-5 * delta_dist:
        break
    self.cut_dist = cut_dist

    # step 2: cal rho
    self.rho = (self.dist_array < self.cut_dist).sum(axis=1)

    # step 3: cal delta & father
    rho_idx = np.argsort(-self.rho)
    father = list(range(len(self.rho)))
    delta = np.ones(len(self.rho)) * (self.max_dist + 1)
    for i in range(1, self.n):
      for j in range(0, i):
        ori_i, ori_j = rho_idx[i], rho_idx[j]
        if self.dist_array[ori_i, ori_j] < delta[ori_i]:
          delta[ori_i] = self.dist_array[ori_i, ori_j]
          father[ori_i] = ori_j
    self.delta = delta
    self.father = father

    print('delta-rho map calculated')

  def do_cluster(self, n_center):
    assert n_center < self.n
    self.label = [x for x in self.father]

    # select center according to gamma
    gamma = self.rho * self.delta
    gamma_idx = np.argsort(-gamma)[:n_center]
    for center in gamma_idx:
      self.label[center] = center

    # union find class center for each point
    for i in range(self.n):
      if self.label[i] == i:
        continue
      passed_by = []
      label_i = i
      while label_i != self.label[label_i]:
        passed_by.append(label_i)
        label_i = self.label[label_i]
      for j in passed_by:
        self.label[j] = label_i

    print('clustering done')
