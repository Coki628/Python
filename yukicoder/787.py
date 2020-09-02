"""
・自力AC
・確率
・設定がシンプルだったからガチャガチャやって通せた感はある。。
・とりあえず、実際の状況を元に、「そうなる確率 / 全体」にできないか考えてみる。
・すると、ここで全体とするべきなのは、本当の全体のうち、
　「実際に裏切り者で、裏切り者判定された」と「実際は違うのに、裏切り者判定された」の2つ。
・これらの確率を出して、足して全体として、前者だけ分子側に置けばいい。
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

P, Q = map(float, input().split())
P /= 100
Q /= 100

ans = (Q*P) / ((1-Q)*(1-P) + Q*P) * 100
print(ans)
