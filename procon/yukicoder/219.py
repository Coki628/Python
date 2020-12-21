"""
・自力AC！
・冪乗、log
・冪乗をlogの世界に持ってって掛け算にする。こういうの誤差こわいねー。
　コンテスト後に誤差許容が追加されたらしく、今回は普通に投げてAC。
"""

import sys
from math import log10, floor

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

N = INT()
for i in range(N):
    a, b = MAP()

    loga = log10(a)
    loga *= b
    cnt10 = int(loga)
    ans = floor(pow(10, loga-cnt10)*10)
    ans1, ans2 = divmod(ans, 10)
    print(ans1, ans2, cnt10)
