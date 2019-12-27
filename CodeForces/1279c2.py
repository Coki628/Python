# -*- coding: utf-8 -*-

"""
・自力ならず。。
・データの持ち方を工夫。各プレゼントがAのどの位置にあるかを持つ。
・どこまで潜ったことがあるかを持っておいて、そこより手前にあるものは
　適切にソート済とみなして、全てコスト1で取り出せるとする。
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

for _ in range(INT()):
    N, M = MAP()
    A = LIST()
    B = LIST()
    A2 = [-1] * (N+1)
    for i, a in enumerate(A):
        # 各プレゼントがAのどの位置にあるかを持つ
        A2[a] = i
    depth = cnt = 0
    for i, b in enumerate(B):
        a = A2[b]
        # 潜ったことがある深さなら、コスト1で取得
        if a < depth:
            cnt += 1
        else:
            # 潜る深さa - もう終わった数i = 実際に取り出す数
            cnt += (a-i) * 2 + 1
            # 最大深さを更新
            depth = a + 1
    print(cnt)
