# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/abc002
・公式解に合わせて愚直
・これで通るならこの方が全然楽だな。。
・0.03秒AC(bitDPは0.05秒)、今回N=12だけど、組み合わせnCrは案外いけるもんだな。
　全組み合わせ調べても2^Nくらいは大丈夫と思っていいんかな。
"""

import sys
from itertools import combinations

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()
G = list2d(N, N, False)
for i in range(M):
    x, y = MAP()
    G[x-1][y-1] = True
    G[y-1][x-1] = True

nodes = range(N)
ans = 1
# i人派閥が作れるかどうか
for i in range(2, N+1):
    # i人になる全組み合わせcomb
    for comb in combinations(nodes, i):
        is_valid = True
        # combから作れる全ペア
        for pair in combinations(comb, 2):
            x, y = pair
            if not G[x][y]:
                is_valid = False
        # 全てのペアが繋がっていれば有効
        if is_valid:
            ans = i
print(ans)
