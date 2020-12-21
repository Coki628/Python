# -*- coding: utf-8 -*-

"""
・正規表現
"""

import sys, re

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

S=input()
N=len(S)

ans=0
# 0文字以上でACGTのいずれかのみ
pattern = re.compile(r'^[ACGT]*$')
for i in range(N):
    for j in range(i+1, N+1):
        if pattern.match(S[i:j]):
            ans=max(ans, j-i)
print(ans)
