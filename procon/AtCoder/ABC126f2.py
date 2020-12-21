# -*- coding: utf-8 -*-

"""
・調査用
・少ない数で全列挙して規則性を探す。
　こういうのを本番中に試しに作ってれば、規則性に気付けたかもしれない。
"""

import sys
from itertools import permutations

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

M,K=MAP()

nums=[]
for i in range(2**M):
    nums.append(i)
    nums.append(i)

ans=set()
for p in permutations(nums):
    ok=True
    for i in range(2**M):
        tmp=i
        for j in range(2**(M+1)):
            if p[j]==i:
                for k in range(j+1, 2**(M+1)):
                    tmp^=p[k]
                    if p[k]==i:
                        break
                if tmp!=K:
                    ok=False
                break
    if ok:
        ans.add(p)
print(*ans)
