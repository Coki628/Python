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

N, X, Y = MAP()

# 簡単のため大小関係を固定
X, Y = min(X, Y), max(X, Y)
# 偶奇が合ってるなら直接会いに行く
if X % 2 == Y % 2:
    ans = abs(X - Y) // 2
# 合わないなら、どっちかの端で調整してから向かう
else:
    cnt1 = abs(X - 1) + 1
    Y1 = Y - cnt1
    cnt2 = abs(Y - N) + 1
    X2 = X + cnt2
    ans = min(cnt1 + abs(1 - Y1) // 2, cnt2 + abs(N - X2) // 2)
print(ans)
