# -*- coding: utf-8 -*-

"""
・蟻本演習3-2-1、さくっと自力AC
・尺取法、条件を満たす最長の区間
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

N, K = MAP()
A = LIST(N)

# 例外処理
if min(A) == 0:
    print(N)
    exit()

# 尺取法
l = r = ans = 0
sm = 1
# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N and sm * A[r] <= K:
        sm *= A[r]
        r += 1
    # 積がK以下になる区間の長さのうち、最大を取る
    ans = max(ans, r - l)
    if l == r:
        # 左が右に追いついたら、右も左に合わせて+1する
        r += 1
    else:
        # 左を動かす分、合計から引く
        sm //= A[l]
    l += 1
print(ans)
