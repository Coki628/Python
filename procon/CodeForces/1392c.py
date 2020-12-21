"""
・自力AC
・上りか下りかを保持しておいて、
　方向が切り替わるタイミングか、maxを更新するタイミングでまとめて足す。
・考慮足りてるか不安だったけど無事AC。
・絶対もっと簡単にできるはず、と思って早い人の見に行ったらやっぱり。
　実は下りに注目すると、下がった差分を全部足せばいい、となる。。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    A = LIST() + [INF]

    stack = []
    mx = A[0]
    lo = A[0]
    ans = 0
    up = True
    for i in range(1, N+1):
        if A[i-1] <= A[i] and up:
            if A[i] >= mx:
                ans += mx - lo
                lo = A[i]
                mx = A[i]
        elif A[i-1] <= A[i] and not up:
            up = True
            lo = A[i-1]
        elif A[i-1] > A[i] and up:
            up = False
            ans += A[i-1] - lo
            lo = A[i-1]
    ans += mx - lo
    print(ans)
