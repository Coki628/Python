# -*- coding: utf-8 -*-

"""
・ギャグ
・操作回数制限が2NMなのに2(N+M)って思い込んでしまって終わった。。。
"""
 
import sys
from collections import Counter
 
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
INF = 10 ** 18
MOD = 10 ** 9 + 7
 
H, W, K = MAP()
for _ in range(K*2):
    MAP()

ans = 'L'*(W-1) + 'U'*(H-1)
for i in range(H):
    if i % 2 == 0:
        ans += 'R'*(W-1)
    else:
        ans += 'L'*(W-1)
    if i != H-1:
        ans += 'D'
print(len(ans))
print(ans)
