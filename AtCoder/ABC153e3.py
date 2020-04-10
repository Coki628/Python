# -*- coding: utf-8 -*-

"""
・ちょっと別の問題(diverta2019_2d)で個数制限なしナップザックについて知見が増えたので。
・ループのネスト逆にして、同じ配列について更新していくとすっきり。
・速度もこっちのが速くて(多分次元が浅いから)、pypyAC0.9→0.6秒。
・ネストの向きそのままでも大丈夫だった。しかもこっちのが速い。0.4秒。
"""

import sys

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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

H, N = MAP()
AB = []
for i in range(N):
    a, b = MAP()
    AB.append((a, b))

dp = [INF] * 20007
dp[0] = 0
for a, b in AB:
    for i in range(20007):
        if i-a >= 0:
            dp[i] = min(dp[i], dp[i-a] + b)
ans = min(dp[H:])
print(ans)
