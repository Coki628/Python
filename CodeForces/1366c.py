"""
・しっかり自力AC
・グリッド、回文
・A~Cでこれが一番スムーズに考察進んですんなり解けたし。。
"""

import sys
from collections import Counter

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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    H, W = MAP()
    grid = [[]] * H
    for i in range(H):
        grid[i] = LIST()

    C1 = Counter()
    C2 = Counter()
    for i in range(H):
        for j in range(W):
            dist1 = i + j
            dist2 = (H-i-1) + (W-j-1)
            # 真ん中は変更不要
            if dist1 == dist2:
                continue
            dist = min(dist1, dist2)
            # 始点or終点からの距離毎にまとめる
            C1[dist] += 1
            if grid[i][j]:
                C2[dist] += 1
    ans = 0
    for k, v in C1.items():
        # 1だった個数と0だった個数の少ない方を変更する
        ans += min(C2[k], v-C2[k])    
    print(ans)
