"""
・さくっと自力AC。じゃんけん関係ない。。
・なんで茶diffかと思ったら、これもC++キラーか。
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

name1, p1, _ = input().split()
name2, p2, _ = input().split()

p1 = int(p1)
p2 = int(p2)
if p1 > p2:
    print(name1)
elif p1 < p2:
    print(name2)
else:
    print(-1)
