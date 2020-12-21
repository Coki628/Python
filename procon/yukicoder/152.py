"""
参考：https://mathwords.net/pitagorasusuu
・自力AC！
・ピタゴラス数
・調べたら求める式は出た。
・とりあえず式の内容的に2乗しているので、√Lくらいまでやれば十分ぽいなー、
　ならば2重ループOKで全探索できそう、となる。
・最終的に同じ比のやつ省くのとかでGCD使ったりもしたから、計算量1000万にlog乗ってやだなー、
　ってなったけど、一応投げてみたら無事pypyAC0.39秒。
"""

import sys
from math import gcd

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**18
MOD = 10**9 + 7
EPS = 10**-10

def calc(m, n):
    a = abs(m**2 - n**2)
    b = 2*m*n
    c = m**2 + n**2
    return [a, b, c]

L = INT()

ans = 0
S = set()
i = 1
while 1:
    j = i + 1
    if sum(calc(i, j))*4 > L:
        break
    while 1:
        res = calc(i, j)
        sm = sum(res)*4
        if sm > L:
            break
        g = 0
        for a in res:
            g = gcd(g, a)
        for k in range(3):
            res[k] //= g
        res = tuple(sorted(res))
        if res not in S:
            ans += 1
            S.add(res)
        j += 2
    i += 1
print(ans)
