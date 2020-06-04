"""
・自力AC
・門松列、構築
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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

N, K = MAP()

# まず完全なやつを作る
a = 0
b = 10 ** 9 // 2
ans = []
for i in range(N):
    if i % 2 == 0:
        ans.append(a)
        a += 1
    else:
        ans.append(b)
        b += 1

# Kの値に応じて門松列を減らす
for i in range(N-K, N):
    ans[i] = ans[i-1]

print(*ans)
