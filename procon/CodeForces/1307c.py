# -*- coding: utf-8 -*-

"""
・自力AC
・結局、最大を取るのは1文字か2文字の場合のみ。
・計算量26*26*10万=6700万つらいかと思ったら、以外とpypyAC0.8秒。
・複数テストケース系の問題は同じ計算量の単体テストケースのやつより通りやすい感がある。
"""

import sys
from collections import Counter
from itertools import accumulate

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
# sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

S = [ord(a)-97 for a in input()]
N = len(S)

acc = list2d(26, N, 0)
for i, s in enumerate(S):
    acc[s][i] += 1
for s in range(26):
    acc[s] = list(accumulate(acc[s]))

C = Counter(S)
# 1文字の場合
mx = max(C.values())
# 2文字の場合をアルファベット全通り試す
for i in range(26):
    cnta = C[i]
    for j in range(26):
        cntb = C[j]
        if i == j:
            # 同じ2文字なら場所に関わらずnC2
            mx = max(mx, cnta * (cnta-1) // 2)
        else:
            cnt = 0
            for k in range(N):
                # 違う文字なら、ある後ろの文字の位置より前に出現した前の文字の数だけ作れる
                if S[k] == j:
                    cnt += acc[i][k]
            mx = max(mx, cnt)
print(mx)
