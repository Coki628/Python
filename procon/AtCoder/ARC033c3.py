# -*- coding: utf-8 -*-

"""
・蟻本演習3-3-4
・平方分割
・計算量はクエリ数*√20万で20万*約450=9000万だけど、更新クエリはO(1)だから実際はもっと少ないはず。
・pythonTLE,pypyAC0.8秒。
"""

import sys
from math import sqrt

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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

N = 2 * 10 ** 5
# 平方分割
D = ceil(sqrt(N))
bucket = [[] for i in range(D)]
ans = []
for i in range(INT()):
    t, x = MAP()
    if t == 1:
        # 各値をD個のバケットに割り振る
        bucket[x//D].append(x)
    else:
        # 添字で探すので0-indexedしておく
        x -= 1
        cnt = i = 0
        # 各バケットの要素数を数えていけば、x番目が格納されている場所が見つかる
        while i < D and cnt + len(bucket[i]) <= x:
            cnt += len(bucket[i])
            i += 1
        bucket[i].sort()
        j = x - cnt
        # バケット内で「x-その手前のバケットまでの要素数」番目の値が答え
        ans.append(str(bucket[i][j]))
        bucket[i].pop(j)
# 空リストで改行のみ出力がWAになるようなので対応する
if ans:
    print('\n'.join(ans))
