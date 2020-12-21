# -*- coding: utf-8 -*-

"""
・次index取得の前計算DPを汎用的なやつにした。
"""

import sys
from collections import Counter

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

for _ in range(INT()):
    
    S = [ord(c) - 97 for c in input()]
    T = [ord(c) - 97 for c in input()]
    N = len(S)
    M = len(T)

    C1 = Counter(S)
    C2 = Counter(T)
    for k, v in C2.items():
        # 例外処理
        if C1[k] == 0:
            print(-1)
            break
    else:
        # nxt[i][c] := i文字目以降で最初に文字cが登場するindex(存在しないときはN)
        nxt = list2d(N+1, 26, N)
        for i in range(N-1, -1, -1):
            for c in range(26):
                nxt[i][c] = nxt[i+1][c]
            nxt[i][S[i]] = i
        cnt = 1
        i = 0
        for c in T:
            # この先にもうない文字だったら最初に戻る
            if nxt[i][c] == N:
                cnt += 1
                i = 0
            # 該当文字の場所(の次)まで飛ぶ
            i = nxt[i][c] + 1
        print(cnt)
