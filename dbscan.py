# -*- coding: utf-8 -*-

import numpy as np

UNCLASSIFIED = False
NOISE = None


def _dist(p, q):
    return np.sqrt(np.power(p-q, 2).sum())


def _eps_neighborhood(p, q, eps):
    return _dist(p, q) < eps


def _region_query(m, point_id, eps):
    n_points = m.shape[1]
    seeds = []
    for i in range(0, n_points):
        if i == point_id:
            continue
        if _eps_neighborhood(m[:, point_id], m[:, i], eps):
            seeds.append(i)
    return seeds


def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(m, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        # 确定一个类
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id

        # 以当前类的邻域为seed，继续扩展
        while seeds:
            current_point = seeds[0]
            # 以邻域内的一个种子为核心，继续向外搜索
            results = _region_query(m, current_point, eps)
            if len(results) >= min_points:
                # results中不可能存在NOISE的数据
                for next_point in results:
                    # 检查当前点是否已经分类
                    # 未分类
                    if classifications[next_point] == UNCLASSIFIED:
                        # 使得搜索继续下去的关键
                        seeds.append(next_point)
                        classifications[next_point] = cluster_id
            # 将搜索过的种子去除
            seeds = seeds[1:]
        return True


def dbscan(m, eps, min_points):
    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1
    return classifications


def test_dbscan():
    m = np.matrix(
        '1 1.2 0.8 1 3.7 3.9 3.6 10 11 12 100; 1.1 0.8 1 1 4 3.9 4.1 10 12 11 99')
    eps = 3
    min_points = 2
    ret = dbscan(m, eps, min_points)
    print(ret)
    # assert ret == [1, 1, 1, 2, 2, 2, None]


if __name__ == '__main__':
    test_dbscan()
