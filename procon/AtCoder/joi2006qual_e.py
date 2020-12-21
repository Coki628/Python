# -*- coding: utf-8 -*-

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

a, b, c = MAP()
M = a + b + c
N = INT()
fail = []
ans = [2] * M
for _ in range(N):
    i, j, k, r = MAP()
    i -= 1; j -= 1; k -= 1
    if r:
        # 先に正常な部品を確定する
        ans[i] = ans[j] = ans[k] = 1
    else:
        fail.append((i, j, k))

for i, j, k in fail:
    # 3つのうち2つが正常と分かっていれば、残る1つが異常
    if ans[i] == ans[j] == 1:
        ans[k] = 0
    if ans[j] == ans[k] == 1:
        ans[i] = 0
    if ans[k] == ans[i] == 1:
        ans[j] = 0

[print(a) for a in ans]
