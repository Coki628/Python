# -*- coding: utf-8 -*-

"""
・終了30分後くらいに自力AC
・条件足りてるかあんま自信なかったけど、ちゃんと通った。
　時間内に通せてたらでかかったのになー。。
・二分探索、マンハッタン距離
・範囲はめちゃくちゃでかい。がしかし考えうる目的地の座標は実は数が少ない。
・とはいえ順番全探索(順列)できる程小さくはないので(多分最大60くらい)、最適な回り方を考える。
・目的座標は単調増加っぽいので、1度どこかに着いたら、あとは左下か右上に向かって順番に進むぽい。
・なので、出発地点にする場所全探索と、いくつ先に行ってから引き返して反対を目指すかで0〜目的個数までを
　左右試して小さい方取り、って感じで、後は目的個数を二分探索。(N小さいから多分線形でもいけたと思う)
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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

x0, y0, ax, ay, bx, by = MAP()
sx, sy, T = MAP()
A = [(x0, y0)]
while A[-1][0] < INF and A[-1][1] < INF:
    xi, yi = A[-1]
    A.append((ax*xi+bx, ay*yi+by))
N = len(A)

dist = [0] * (N-1)
for i in range(N-1):
    x1, y1 = A[i]
    x2, y2 = A[i+1]
    dist[i] = abs(x2 - x1) + abs(y2 - y1)

def check(m):
    res = INF
    for i in range(N):
        # 最初に行く座標
        init = abs(A[i][0] - sx) + abs(A[i][1] - sy)
        if init > T:
            continue
        for l in range(m):
            r = m - l - 1
            if i - l < 0 or i + r > N:
                continue
            # 左にl個行って戻って右にr個 or 右にr個行って戻って左にr個 の小さい方
            add = min(sum(dist[i-l:i])*2 + sum(dist[i:i+r]), sum(dist[i-l:i]) + sum(dist[i:i+r]*2))
            res = min(res, init + add)
    return res <= T

res = bisearch_max(0, N+1, check)
print(res)
