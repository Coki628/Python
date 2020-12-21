"""
・蟻本演習4-2-4
・Nim
・問題文見て、よっしゃNimやるだけ、って思ったら制約でかかった。
・1から1ずつ増える数列がビットで見てどうなるか観察していくと、3と7でxorが0になって、
　あ、これxorがmod 4で周期になってるやつだ、って思って投げてAC。お気持ち。。。
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

N = INT()

if N % 4 != 3:
    print('O')
else:
    print('X')
