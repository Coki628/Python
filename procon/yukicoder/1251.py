"""
・なんとか自力AC！
・三分探索、中央値
・こういうのは凸関数っぽいので、三分探索しようとする。WA。合わない。。
・数列Bでの重み付けがなければ中央値になると思う。でも、重みが付いても要領は同じで、
　結局Aのいずれと同値にするのが一番適切になりそう、となる。
・広い範囲を探索しようとしてたけど、Aのいずれかと同値になるなら、数列Aソートして、
　Aに含まれる値だけを調べればいいじゃないか、となる。
・それでもWA。。よく考えたら、res[0]とres[1]しか調べてなかったけど、
　lo+2 < hiだから間にもう1つあるじゃん。。となる。
・それを踏まえてもWA。。もう雑に範囲広げよう、と前後10ずつくらいさらにチェック。これでAC。
・嘘解じゃないか心配。。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def trisearch_min(lo, hi, func):
    """ 三分探索 """

    m1 = lo
    m2 = hi
    while lo+2 < hi:
        m1 = (lo*2+hi) // 3
        m2 = (lo+hi*2) // 3
        res1 = func(m1)
        res2 = func(m2)
        if res1 <= res2:
            hi = m2
        else:
            lo = m1
    return m1, m2

N = INT()
A = LIST()
B = LIST()

def check(m):
    res = 0
    for i in range(N):
        res += B[i] * abs(A2[m]-A[i])
    return res

A2 = sorted(A)
l, r = trisearch_min(0, N, check)
mn = INF
for i in range(max(l-10, 0), min(r+10, N)):
    res = check(i)
    if res < mn:
        mn = res
        ans = (A2[i], res)
print(*ans)
