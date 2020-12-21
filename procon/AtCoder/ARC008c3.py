# -*- coding: utf-8 -*-

"""
参考：https://atcoder.jp/contests/arc008/submissions/7066923
・ダイクストラ応用、メモリ管理効率化(笑)
・偉大な先人の方々の提出を参考にしてようやくAC。
・以下、得られた知見を列挙。
　・使い終わったリストをdelやってる人を何人か見たけど、自分ではほぼ効果が見られなかった。
　・上部の定形パートを不要なの全部消して試したけど、これも効果はなかった。
　・普通のlistで値整えてnumpy系に変換してたけど、これを最初からnumpyでやったのが一番メモリには効果があった。
　・csr_matrixをdijkstraに渡してる例が多いようだったけど、今回はnumpy配列(ndarray)のが速かった。
"""

import sys
if sys.version_info.minor >= 5: from math import gcd
else: from fractions import gcd
from math import hypot

import numpy as np
from scipy.sparse.csgraph import dijkstra

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
XYTR = []
for i in range(N):
    x, y, t, r = MAP()
    XYTR.append((x, y, t, r))

# 全ペアについてコスト計算して辺を張る
G = np.zeros((N, N), dtype=np.float32)
for i in range(N):
    x, y, t, r = XYTR[i]
    for j in range(i+1, N):
        x2, y2, t2, r2 = XYTR[j]
        dist = hypot(x-x2, y-y2) / min(t, r2)
        G[i,j] = dist
        dist = hypot(x-x2, y-y2) / min(t2, r)
        G[j,i] = dist

# 例外処理
if N == 1:
    print(0)
    exit()

# 自分からの最短距離
res = list(dijkstra(G, indices=0))

res.sort(reverse=True)
ans = 0
for i in range(N-1):
    # 1秒1回しか投げれないので、後回しにする程コストとして秒数が乗る
    ans = max(ans, res[i] + i)
print(ans)
