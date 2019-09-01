# -*- coding: utf-8 -*-

"""
・結構さくっと400点自力AC！
・回文系
"""

import sys
from collections import Counter

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

S = input()

C = Counter(S)
# 一番大きいまとまりを1つ作る
a = 0
first_odd = True
ans = 0
for k, v in C.most_common():
    # 偶数は全てまとめられる
    if v%2 == 0:
        a += v
    else:
        # 奇数は1回だけ入れられるので、最大のものを入れる
        if first_odd:
            a += v
            first_odd = False
        else:
            # 残りは1つ減らして偶数として入れる
            a += v - 1
            # 減らした分の1つをスコアに追加
            ans += 1
# 作った大きいやつのスコアを足す
ans += a ** 2
print(ans)
