# -*- coding: utf-8 -*-

"""
・条件整えたつもりだったけどWAだったので、全探索にした。
"""

import sys

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

# 時刻の比較関数
def comp(hm1, hm2):
    if hm1[0] < hm2[0]:
        return True
    elif hm1[0] == hm2[0] and hm1[1] <= hm2[1]:
        return True
    else:
        return False

# 時計を反転させる操作
def rev(h, m):
    return ((h+6)%12, (m+30)%60)

limit = LIST()
cur = LIST()

# 考えうる時刻を全部調べる
for h in range(12):
    for m in range(60):
        # この時刻が現在時刻以降なら使える
        if comp(cur, (h, m)):
            # 時計の反転
            revhm = rev(h, m)
            # 間に合うかどうか
            if comp(revhm, limit):
                Yes()
                exit()
# 全部試してダメならNG
No()
