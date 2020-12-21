"""
・自力AC
・優先度付きキュー
・なるべく小さい方から貪欲にやる。切り捨て除算で1WA。半分にした時小数にもなる。
"""

import sys
from heapq import heapify, heappush, heappop

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

x, y, h = MAP()

x *= 1000
y *= 1000
que = [x, y]
heapify(que)
ans = 0
while que:
    cur = heappop(que)
    if cur > h:
        h *= 2
        cur /= 2
        heappush(que, cur)
        ans += 1
print(ans)
