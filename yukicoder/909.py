"""
・自力AC
・割とスムーズに想定解の考察にいけたのでいい感じ。
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
EPS = 10 ** -10

N = INT()
X = LIST()
Y = LIST()

# 一番短いやつに合わせる
mn = INF
for i, (x, y) in enumerate(zip(X, Y)):
    mn = min(mn, x + y)

ans = [0] * N
for i in range(N):
    # xで目一杯伸ばして、yは適当に短くすればいい(0でもいいので)
    ans[i] = min(X[i], mn)
ans = [0] + ans + [mn]
print(mn)
for a in ans:
    print(a)
