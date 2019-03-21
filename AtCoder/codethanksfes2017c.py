# -*- coding: utf-8 -*-

"""
・300点速攻自力AC
・優先度付キュー
・10万の(N+K)logNとはいえ、もうちょいかかると思ったけど意外とpython0.5秒AC
"""

import sys
from heapq import heappop, heappush, heapify

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

N,K=MAP()
AB=[[0]*2 for i in range(N)]
for i in range(N):
    AB[i][0],AB[i][1]=MAP()
heapify(AB)

cnt=time=0
while cnt<K:
    a,b=heappop(AB)
    cnt+=1
    time+=a
    a+=b
    heappush(AB, [a,b])
print(time)
