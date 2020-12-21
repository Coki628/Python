# -*- coding: utf-8 -*-

"""
・再帰探索
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

m, n, N = MAP()

ans = 0
def rec(cur, mod):
    global ans
    ans += cur
    # 今回回収分から作れる分
    nxt = cur // m * n
    # 今回作った余りと前回までの余りから作れる分
    nxt += (mod+cur%m) // m * n
    # 1本も作れなくなったら終了
    if nxt == 0:
        return
    # 今回新たに作った分と余りを次の処理に送る
    rec(nxt, (mod+cur%m)%m)
    return

rec(N, 0)
print(ans)
