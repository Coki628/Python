# -*- coding: utf-8 -*-

"""
・D自力AC！
・インタラクティブ問題
・木の直径、前やったの思い出せて良かった。
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

# まず適当な頂点から一番遠いところを探す
mx=a=0
for i in range(2, N+1):
    print('? {0} {1}'.format(1, i))
    sys.stdout.flush()
    dist=INT()
    if dist>mx:
        mx=dist
        a=i

# その一番遠かったところから見て、再度一番遠いところを探す
ans=mx
for i in range(1, N+1):
    print('? {0} {1}'.format(a, i))
    sys.stdout.flush()
    dist=INT()
    ans=max(ans, dist)

print('! {0}'.format(ans))
