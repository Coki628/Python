"""
・自力AC
・実験する。すると、0に戻ってくるケースはそう多くなさそう、となる。
・むしろもしかするとサンプルにある1になるやつと2になるやつ以外ないんじゃないか、となる。
・念のため適当に10回くらいまでループ回して、終わったらそこ、終わらなければ-1、をやってみる。無事AC。
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

a, b = MAP()

n = 0
for i in range(1, 11):
    n = a*n + b
    if n == 0:
        print(i)
        exit()
print(-1)
