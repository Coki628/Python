"""
参考：https://www.try-it.jp/chapters-4962/sections-5093/lessons-5098/
・自力ACだけどちょっとウッってなるやつ。
・無限小数
・「無限小数　見分け方」とかってググると情報が出る。分母の素因数が2,5だけならいいとのこと。
・はいWA。なんでだろう。しばし悩んで気付く。約分してからやんないとダメじゃん。無事AC。
"""

import sys
from math import gcd

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

a, b = MAP()

g = gcd(a, b)
a //= g
b //= g

for x in [2, 5]:
    while b % x == 0:
        b //= x
if b != 1:
    Yes()
else:
    No()
