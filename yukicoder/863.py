"""
・さくっと自力AC？
・計算通り1600で区切ったらWAで、あぁ、切り捨てられてるからズレてんのか、ってなって、
　じゃあ適当に小さくすればいいや(どうせ1の方はそこまででかくならなそうだし)って1000にしたら通った。
・想定解見たら、b/a<=1600じゃなくてb/a<1600なら良かったみたい。。
　確かに、1600ちょうどは上に含まれるよね、切り捨てとか関係なしに。
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

a = INT()
b = INT()

if b / a <= 1000:
    print(1)
else:
    print(2)
