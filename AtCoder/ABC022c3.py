# -*- coding: utf-8 -*-

"""
組み合わせ * ダイクストラ => N^2 * N^2 => O(N^4)かかる
ワーシャルフロイド + 組み合わせ => N^3 + N^2 => O(N^3)にする
"""

from itertools import combinations
from scipy.sparse.csgraph import floyd_warshall

N,M = map(int, input().split())
# ワーシャルフロイド用に、2頂点と距離を保持する2次元リスト: [頂点1][頂点2] = 距離
fw = [[float('inf')] * N for i in range(N)]
# 1につながる道を格納するリスト: [(1とつながる頂点, そこまでの距離), ...]
from1 = []

for i in range(M):
    u,v,l = map(int, input().split())
    # 1につながる道
    if u == 1:
        from1.append((v-1, l))
    elif v == 1:
        from1.append((u-1, l))
    # それ以外はワーシャルフロイド用
    else:
        # 無向グラフなのでどっち向きも同じ
        fw[u-1][v-1] = l
        fw[v-1][u-1] = l
# 無向グラフなのでdirected=Falseで
fw = floyd_warshall(fw, directed=False)
# 1につながる道2つの全組み合わせ
comb = list(combinations(from1, 2))
# 各組み合わせのうち最小を取得する
ans = float('inf')
# pair = [(1とつながる頂点, そこまでの距離), (1とつながる頂点, そこまでの距離)]
for pair in comb:
    # ワーシャルフロイドで求めた2点間の距離 + その2点から1への距離
    ans = min(fw[pair[0][0]][pair[1][0]] + pair[0][1] + pair[1][1], ans)

# 該当ルートがなかった時の例外処理
if ans == float('inf'):
    print(-1)
else:
    print(int(ans))
