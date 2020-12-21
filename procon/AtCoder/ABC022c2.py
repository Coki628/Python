# -*- coding: utf-8 -*-

"""
組み合わせ * ダイクストラ => N^2 * N^2 => O(N^4)かかる
TLE確定なんだけど練習用
"""

from itertools import combinations
from scipy.sparse.csgraph import dijkstra

N,M = map(int, input().split())
# ダイクストラ用に、2頂点と距離を保持する2次元リスト: [頂点1][頂点2] = 距離
dk = [[float('inf')] * N for i in range(N)]
# 1につながる道を格納するリスト: [(1とつながる頂点, そこまでの距離), ...]
from1 = []

for i in range(M):
    u,v,l = map(int, input().split())
    # 1につながる道
    if u == 1:
        from1.append((v-1, l))
    elif v == 1:
        from1.append((u-1, l))
    # それ以外はダイクストラ用
    else:
        # 無向グラフなのでどっち向きも同じ
        dk[u-1][v-1] = l
        dk[v-1][u-1] = l

# 1につながる道2つの全組み合わせ
comb = list(combinations(from1, 2))
# 各組み合わせのうち最小を取得する
ans = float('inf')
# pair = [(1とつながる頂点, そこまでの距離), (1とつながる頂点, そこまでの距離)]
for pair in comb:
    # indicesで始点(pair[0][0])を固定、返されるリストから終点の位置([pair[1][0]])を選ぶ
    distance = dijkstra(dk, directed=False, indices=pair[0][0])[pair[1][0]]
    # ダイクストラで求めた2点間の距離 + その2点から1への距離
    ans = min(distance + pair[0][1] + pair[1][1], ans)

# 該当ルートがなかった時の例外処理
if ans == float('inf'):
    print(-1)
else:
    print(int(ans))
