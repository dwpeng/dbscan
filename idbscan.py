from typing import Union, List, Tuple, Set, Iterable
import numpy as np


class Status:
    UNCLASSIFIED = -1
    NOISE = -2
    KERNEL = -3
    CLASSIFIED = -4


class ClusterID:
    """分类id

    这里使用类，是为了可以随时在程序任意处更新id值"""
    _id = 1

    @classmethod
    def update_id(cls):
        cls._id += 1

    @classmethod
    @property
    def id(cls):
        return cls._id

    @classmethod
    def reset_id(cls, id):
        cls._id = id

    @classmethod
    def __str__(cls):
        return 'ID(%d)' % cls._id


def _dist(p:np.ndarray, q:np.ndarray) -> np.number:
    """ 计算p与q的距离 """
    return np.sqrt(np.power(p-q, 2).sum())


def _eps_neighborhood(p:np.ndarray, q:np.ndarray, eps: Union[float, int]) -> bool:
    """ 判断p与q的距离是否符合要求 """
    return _dist(p, q) < eps  #type: ignore


def _region_query(
    points:np.ndarray,
    point_id: int,
    eps: Union[float, int],
) -> Tuple[List[int], int]:
    """ 给定一个数据，在所有数据中搜索符合条件的种子 """
    n_points = points.shape[1]
    count = 0
    seeds = []

    for i in range(n_points):
        if _eps_neighborhood(points[:, point_id], points[:, i], eps):
            seeds.append(i)
            count += 1
    return seeds, count


def _nearist_point(
    points:np.ndarray,
    classifications:np.ndarray,
    seeds: List[int],
    point_id: int
) -> int:
    """ 在points中，除seeds中的点之外，距离point最近的一点，即q点 """

    min_point_id = 999999999
    min_dest = 999999999
    for i in range(points.shape[1]):
        if i in seeds:
            continue

        d = _dist(
            points[:, i],
            points[:, point_id]
        )

        if d <= min_dest:
            min_point_id = i
            min_dest = d

    if min_point_id == 999999999:
        return -1

    return min_point_id


def _set_classifications(
    classifications:np.ndarray,
    seeds: List[int],
    cluster_id: Union[ClusterID, int]
):
    if type(cluster_id) == ClusterID:
        cluster_id = cluster_id.id

    for s in seeds:
        classifications[s] = cluster_id


def _check_comm_seed(
    points:np.ndarray,
    comm_seeds: Iterable[int],
    min_point_count: int,
    eps:  Union[float, int]
):
    """ 寻找相同邻域重合部分的点 """
    need_merge_seeds = list()
    s = Status.NOISE
    for pq in comm_seeds:
        pq_seed, pq_seeds_count = _region_query(points, pq, eps)
        if pq_seeds_count >= min_point_count:
            # 将comm_seeds全部设置为p类
            s = Status.KERNEL
            # 将pq的邻域也同时更新
            need_merge_seeds.append((pq, pq_seed))

    return s, need_merge_seeds


def _update_cluster_id(
    classifications:np.ndarray,
    comm_kernel_cluster_seeds: Set[int],
    cluster_id: ClusterID
) -> List[int]:
    """ 查看当前的cluster_id与邻域内核心点对应的cluster_id """
    # 检查区域内部的邻域节点

    comm_area_cluster_id = classifications[list(comm_kernel_cluster_seeds)]

    comm_area_cluster_id_set = set(
        comm_area_cluster_id[comm_area_cluster_id > 0]
    )

    # 获取最小的id值
    need_change_cluster_id = min(comm_area_cluster_id_set)

    if need_change_cluster_id < cluster_id.id:
        # 对p类也进行合并
        comm_area_cluster_id_set.add(cluster_id.id)
        # 确定新的类
        cluster_id.reset_id(need_change_cluster_id)

    # 将其他的类进行合并
    need_merge = []
    # 搜索需要合并的point
    for clustered_id in comm_area_cluster_id_set:
        need_merge.extend(np.argwhere(classifications == clustered_id))

    return need_merge


def _resolve_comm_area(
    points:np.ndarray,
    classifications:np.ndarray,
    comm_seed: List[Tuple[int, List[int]]],
    qseeds: List[int],
    pseeds: List[int],
    cluster_id: ClusterID
):
    """ 处理重合邻域存在核心点的情况 """
    need_merge = set()
    qseeds = set(qseeds) # type: ignore
    # 查看重合邻域内的核心点，其核心点的邻域是否与qseed重合区域内也存在核心点
    for comm_area_kernel_id, comm_area_seed in comm_seed:
        comm_comm_seed_id = set(comm_area_seed) & qseeds # type: ignore
        if not comm_comm_seed_id:
            continue
        else:
            # 存在重合
            # 检查是否存在核心点
            if comm_area_kernel_id in comm_comm_seed_id:
                need_merge.update(comm_area_seed)
                continue
            
            s, comm_seed = _check_comm_seed(
                    points, 
                    comm_area_seed,
                    min_point_count,
                    eps
                )
            if s != Status.KERNEL:
                continue
            for seed_id, seed in comm_seed:
                need_merge.update(seed)

    # 进行更新cluster_id
    need_merge = _update_cluster_id(
        classifications,
        need_merge,
        cluster_id
    )

    # 首先将qseed全部用p的类标记
    _set_classifications(classifications, qseeds, cluster_id)
    _set_classifications(classifications, need_merge, cluster_id)
    _set_classifications(classifications, pseeds, cluster_id)

def _expand_cluster(
    points:np.ndarray,
    classifications:np.ndarray,
    cluster_id: ClusterID,
    point_id: int,
    eps: Union[float, int],
    min_point_count: int
) -> bool:
    """ 根据条件扩展区域 """

    seeds, seeds_count = _region_query(
        points,
        point_id=point_id,
        eps=eps
    )

    # 噪声点
    if seeds_count < min_point_count:
        _set_classifications(classifications, [point_id], Status.NOISE)
        return False

    # 当前point p是一个核心点
    # 首先将，当前种子内部的点都设置为cluster_id类
    # kseed -> kernel seed
    _set_classifications(classifications, seeds, cluster_id)

    # 在邻域外找一点，q，检查q点的情况
    # q点是邻域外距离当前核心点最近的一个点
    q = _nearist_point(points, classifications, seeds, point_id)
    if q == -1:
        return True

    qseeds, qseeds_count = _region_query(
        points=points,
        point_id=q,
        eps=eps
    )

    pq_comm_seed = set(qseeds) & set(seeds)

    # 检查q是否是核心点
    # q不是核心点
    if qseeds_count < min_point_count:
        # 检查是否q的seed中与p的seeds是否存在重合

        # Cond1: 不是核心点，但是有重合
        if pq_comm_seed:
            # 存在重合，检查重合部分是否存在核心点
            s, comm_seed = _check_comm_seed(
                points=points,
                comm_seeds=pq_comm_seed,
                min_point_count=min_point_count,
                eps=eps
            )

            # 重叠区域不存在核心点
            if s == Status.NOISE:
                _set_classifications(classifications, qseeds, Status.NOISE)
            else:
                # 存在核心点
                # 检查邻域内部的核心点类别
                # 更新当前的qseed和共同邻域内的节点为p类
                _resolve_comm_area(
                    points=points,
                    classifications=classifications,
                    comm_seed=comm_seed,
                    qseeds=qseeds,
                    pseeds=seeds,
                    cluster_id=cluster_id
                )

            return True
        else:
            # 不存在重合
            # q的seed暂时全部设置为NOISE
            _set_classifications(classifications, qseeds, Status.NOISE)
            return True

    # q是核心点
    # 检查p和q的邻域是否存在重合
    # 不存在重合
    if not pq_comm_seed:
        # 为qseeds建立新类
        cluster_id.update_id()
        _set_classifications(classifications, qseeds, cluster_id)
        return True

    # 邻域存在重合
    # 检查重合部分是否存在核心点

    # comm_seed为qseed中的核心点
    s, comm_seed = _check_comm_seed(
        points=points,
        comm_seeds=pq_comm_seed,
        min_point_count=min_point_count,
        eps=eps
    )

    # 邻域重合位置存在核心点
    if s == Status.KERNEL:

        _resolve_comm_area(
            points=points,
            classifications=classifications,
            comm_seed=comm_seed,
            qseeds=qseeds,
            pseeds=seeds,
            cluster_id=cluster_id
        )

        return True

    # 重合部分全部都是非核心点
    cluster_id.update_id()
    # 将离q的距离小于离p中最近核心点距离的点
    # 用新的一个类q来标记
    # 获取当前的核心点
    kernel_point = q
    # 获取全部的重叠点
    comm_seed = set(qseeds) & set(seeds)
    # 获取仅在qseed中存在的点
    qseeds = [i for i in qseeds if i not in comm_seed]
    # 开始对comm_seed进行部分更新
    # 情况五
    # 计算comm_seed中距离p中最近核心点的距离
    min_dist = min([
        _dist(
            points[:, i],
            points[:, point_id]
        )
        for i in comm_seed
    ]) # type: ignore

    need_update_comm_seed = [
        i for i in comm_seed
        if _dist(
            points[:, i],
            points[:, kernel_point]
        ) <= min_dist
    ]

    _set_classifications(
        classifications=classifications,
        seeds=need_update_comm_seed + qseeds,
        cluster_id=cluster_id
    )

    return True


def idbscan(
    m:np.ndarray,
    sort_dim: int,
    eps: Union[float, int],
    min_point_count: int
) -> np.ndarray:
    """ IDBSCAN算法实现 """

    # Step1 先按照m中的某一个维度进行排序
    assert sort_dim < m.shape[0], \
    '排序维度选择错误，数据有%s个维度，但是你给了第%d个维度' % (
        '、'.join([str(i) for i in range(m.shape[0])]),
        sort_dim
    )

    mlen = m.shape[1]

    # sort = np.argsort(m[sort_dim, :])
    # m[:] = m[:, sort].reshape(*m.shape)

    # 初始化分类列表
    classifications = np.zeros(mlen, dtype=np.int32) + Status.UNCLASSIFIED

    # 初始化分类id
    cluster_id = ClusterID()

    # Step2 先从已经排序后的m中选择排序索引最小的p开始
    for point_id in range(mlen):
        if classifications[point_id] in [Status.UNCLASSIFIED, Status.NOISE]:
            if _expand_cluster(
                points=m,
                classifications=classifications,
                cluster_id=cluster_id,
                point_id=point_id,
                eps=eps,
                min_point_count=min_point_count
            ):
                cluster_id.update_id()

    # return classifications[np.argsort(sort)]
    return classifications


if __name__ == '__main__':
    m = np.matrix(
        '1 1.2 0.8 1 3.7 3.9 3.6 10 11 12 100; 1.1 0.8 1 1 4 3.9 4.1 10 12 11 99')
    eps = 1.2
    min_point_count = 8
    ret = idbscan(m, 1, eps, min_point_count)
    print(ret)
