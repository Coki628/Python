# -*- coding: utf-8 -*-

"""
・結構かかったけど500点自力AC！
・二部グラフかどうかで場合分けって気付けたのは嬉しい。
"""

import sys
from collections import Counter

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

# 1からnまでの等差数列の和
def get_sum(n):
    return (1+n)*n//2

N,M=MAP()
nodes=[[] for i in range(N)]
for i in range(M):
    a,b=MAP()
    nodes[a-1].append(b-1)
    nodes[b-1].append(a-1)

visited=[False]*N
group=['']*N
# 二部グラフかどうかを判定する
def rec(u):
    if visited[u]:
        return True
    visited[u]=True
    for v in nodes[u]:
        if group[u]==group[v]:
            return False
        if group[u]=='a':
            group[v]='b'
        elif group[u]=='b':
            group[v]='a'
        if not rec(v):
            return False
    return True

group[0]='a'
if rec(0):
    C=Counter(group)
    # 二部グラフなら各グループの数に応じた全組み合わせ-元々ある辺
    ans=C['a']*C['b']-M
else:
    # そうでなければ全辺の組み合わせ-元々ある辺
    ans=get_sum(N-1)-M
print(ans)
