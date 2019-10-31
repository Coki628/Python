# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-1、さくっと自力AC
・尺取法、条件を満たす区間の組の数え上げ
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
INF = float('inf')
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

# 尺取法
l = r = 0
# l == r は常に条件を満たすので、予めNは足しておく
ans = N
# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N-1 and A[r] < A[r+1]:
        r += 1
    # 単調増加な区間の分、合計に足す
    ans += r - l
    if l == r:
        # 左が右に追いついたら、右も左に合わせて+1する
        r += 1
    else:
        pass
    l += 1
print(ans)
