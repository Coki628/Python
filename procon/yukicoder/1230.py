"""
参考：https://drive.google.com/file/d/10QS8QDto4KoB75her-zAPmgyVmluvau2/view
・自力AC、ってかサンプルエスパーしただけ。。
・確率、モンティ・ホール問題
・解説読んだ。こういうのがあるのか。
　何選んだって1/2に思えるね。2/3になるのはほんとに非自明だと思う。
"""

import sys

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

pqr = LIST()

pqr.sort()
ans = (pqr[2]+pqr[1]) / sum(pqr)
print(ans)
