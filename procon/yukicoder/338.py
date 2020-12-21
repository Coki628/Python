"""
・自力ならず。。粘ったから悔しいな。
・四捨五入、誤差
・色々試行錯誤してもテストケース1つだけ通らなくて、結局解答と人のコード見てしまった。
・結局原因は、四捨五入の時じゃなくて、その前の除算の時の誤差だった。。そこをDecimalでやったらAC。
・四捨五入をintでやる方も試して、除算がDecimalならそっちもACしたけど、そっちのがちょっと遅かった。
・計算量は500^2=25万だけど、pythonAC0.3秒、pypyTLE。。pypyのDecimalは使い物にならないみたいだな。。
"""

import sys
from decimal import Decimal, ROUND_HALF_UP

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

A, B = MAP()
if A > B:
    A, B = B, A

ans = INF
for x in range(500):
    x = Decimal(x)
    for y in range(int(x), 500):
        if x == y == 0:
            continue
        y = Decimal(y)
        a = (x / (x+y) * 100).quantize(Decimal('0'), ROUND_HALF_UP)
        b = (y / (x+y) * 100).quantize(Decimal('0'), ROUND_HALF_UP)
        if a == A and b == B:
            ans = min(ans, x+y)
print(ans)
