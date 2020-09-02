"""
・結構すんなり自力AC！
・エラトステネス応用
・題意のようなグラフは、自分の2倍の数が出てきた時に他と繋がる、みたいな形になる。
　つまり「孤立した素数たち」と「連結になったその他全部」の2つのグループになる。
・素数列挙っぽく連結になった場所を埋めていき、最後に家Pが上記どっちのグループに属するか確認すればいい。
　前者なら1、後者なら連結になった成分全部、が答え。
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

N, P = MAP()

ok = [0] * (N+1)
for i in range(2, N+1):
    if not ok[i] and i*2 <= N:
        for j in range(i, N+1, i):
            ok[j] = 1

if ok[P]:
    ans = sum(ok)
    print(ans)
else:
    print(1)
