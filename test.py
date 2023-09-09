import matplotlib.pyplot as plt
from idbscan import idbscan
from dbscan import dbscan
import pandas as pd
import numpy as np



def test(
    df: pd.DataFrame,
    sort_dim: int,
    eps: float,
    min_point_count,
    use_idbscan = True
):
    points = np.array([
        df['a'],
        df['b']
    ])


    if use_idbscan:
        classifications = idbscan(
            points,
            sort_dim,
            eps,
            min_point_count
        )
    else:
        classifications = dbscan(
            points,
            eps,
            min_point_count
        )

    color_col = []
    color = {}
    for _, group in enumerate(classifications):
        if group is None:
            color_col.append('black')
            continue
        if group < 0:
            color_col.append('black')
            continue

        if group not in color:
            color[group] = f'C{int(group)}'

        color_col.append(color.get(group))


    data = pd.DataFrame({
        'x': points[0, :].flatten(),
        'y': points[1, :].flatten(),
        'class': classifications,
        'c': color_col
    })

    data.to_csv('data.csv', index=None)

    print(set(data['c']))

    plt.scatter(data=data, x='x', y='y', c='c')
    plt.savefig('test.png', dpi=500)

    return classifications


if __name__ == '__main__':
    data = pd.read_csv('data/compound.txt', sep='\t', names=['a', 'b'])
    data = pd.read_csv('data/788.txt', names=['a', 'b'])
    data = pd.read_csv('data/jain.txt', names=['a', 'b', 'c'])

    eps = 2.234
    min_point_count = 20
    c = test(data, 0, eps, min_point_count, use_idbscan=True)
