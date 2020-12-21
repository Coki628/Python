"""
・さくっと自力AC
・左右に行って帰ってするやつ。ソートして、連続するN個を拾う場合を全部見る。
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

N, M = MAP()
A = LIST(M)

A.sort()
ans = INF
for i in range(M-N+1):
    j = i + N - 1
    l, r = A[i], A[j]
    if (l < 0) ^ (r < 0):
        # 正負が分かれていたら、短い方を往復して長い方へ進むとする
        ans = min(ans, min(abs(l), abs(r))*2 + max(abs(l), abs(r)))
    else:
        # 正負が片方ならそっち側に進むだけ
        ans = min(ans, max(abs(l), abs(r)))
print(ans)
