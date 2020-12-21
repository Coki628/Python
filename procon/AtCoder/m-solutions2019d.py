# -*- coding: utf-8 -*-

"""
・500点自力AC
・木の探索
・DFS
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N=INT()
nodes=[[] for i in range(N)]
for i in range(N-1):
    a,b=MAP()
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)
C=LIST()

C.sort(reverse=True)

ans=[0]*N
cnt=0
def rec(cur):
    global cnt
    if ans[cur]:
        return
    ans[cur]=C[cnt]
    cnt+=1
    for nxt in nodes[cur]:
        rec(nxt)
rec(0)

if N==1:
    print(0)
else:
    print(sum(C[1:]))
print(*ans)
