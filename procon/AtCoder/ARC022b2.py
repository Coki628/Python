# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/arc022
・一応、公式解に合わせて尺取法もやっておく。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
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

N = INT()
A = LIST()

ans = l = r = 0
cnt = set()
# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N and A[r] not in cnt:
        cnt.add(A[r])
        r += 1
    ans = max(ans, r-l)
    if l == r:
        # 左が右に追いついたら、右も左に合わせて+1する
        r += 1
    cnt.remove(A[l])
    l += 1

print(ans)
