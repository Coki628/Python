# -*- coding: utf-8 -*-

"""
・ダメだ…WAだ。
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

R, G, B = MAP()
# 大小を統一しておく
R, B = min(R, B), max(R, B)

# 1からnまでの等差数列の和
def get_sum(n): return (1+n)*n//2

def count(a):
    res = 0
    add = 1
    for i in range(1, a):
        res += add
        # 2回毎に加算される数が1増えていく
        if i%2 == 0:
            add += 1
    return res

# どのくらい広がるか
Rl = -100 - ceil((R-1), 2)
Rr = -100 + (R-1)//2
Gl = -(G-1)//2
Gr = ceil((G-1), 2)
Bl = 100 - (B-1)//2
Br = 100 + ceil((B-1), 2)
# どのくらい重複するか
dup1 = max(0, Rr - Gl)
dup2 = max(0, Gr - Bl)

print(count(R)+count(G)+count(B)+dup1**2+dup2**2)
