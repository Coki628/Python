# -*- coding: utf-8 -*-

"""
・600点自力AC！これは嬉しい。
・構築系
・終了状態から、どう戻っていけばいいか考える。
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
INF = float('inf')
MOD = 10 ** 9 + 7

K = INT()

# 数列の最小値
start = K // 50
# 1箇所除く位置
avoid = start + 50 - K % 50

ans = []
for i in range(start, start+51):
    if i != avoid:
        ans.append(i)

print(len(ans))
print(*ans)
