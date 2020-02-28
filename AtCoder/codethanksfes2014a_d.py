# -*- coding: utf-8 -*-

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

N, Q = MAP()
ans = []
for i in range(Q):
    l1, r1, l2, r2 = MAP()
    val = r2 - l2
    # 4通り場合分け
    if l1 <= l2 and r1 <= r2:
        val -= max(r1 - l2, 0)
    elif l1 > l2 and r1 > r2:
        val -= max(r2 - l1, 0)
    elif l1 > l2 and r1 <= r2:
        val -= r1 - l1
    elif l1 <= l2 and r1 > r2:
        val = 0
    ans.append(str(val * 100))
print('\n'.join(ans))
