# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/abc134/editorial.pdf
　　　https://www.youtube.com/watch?v=-o4r74eJzV4
・後ろからLIS(最長増加部分列)
"""

import sys
from bisect import bisect_right

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
A=[INT() for i in range(N)]

def LIS(A: list):
    L = [A[0]]
    for a in A[1:]:
        if a >= L[-1]:
            # aがLの末尾以上なら増加部分列を延長する(今回は広義(=を含む))
            L.append(a)
        else:
            # そうでなければ、「a以下の最大要素の次」をaにする
            # 該当位置は、二分探索で特定できる
            L[bisect_right(L, a)] = a
    return len(L)

# 後ろからLIS
print(LIS(A[::-1]))
