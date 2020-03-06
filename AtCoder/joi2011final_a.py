# -*- coding: utf-8 -*-

"""
・自力AC
・真ん中の文字Oを軸にしてJとIの数を数えに行く。
　対象が3つあって真ん中固定、は結構よくある気がするから、ちゃんと対応できて良かった。
・100万の文字列で制約が1秒。pythonAC0.8秒、pypyAC0.3秒。
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

S = input()
N = len(S)

i = 0
ans = 0
while i < N:
    # Oを軸にしてJとIを見に行く
    if S[i] == 'O':
        j = i
        cnto = 0
        while j < N and S[j] == 'O':
            j += 1
            cnto += 1
        cnti = 0
        while j < N and S[j] == 'I':
            j += 1
            cnti += 1
        # 次回外ループ再開位置を、Iが終わった後まで進める
        nxt = j
        j = i - 1
        cntj = 0
        while j >= 0 and S[j] == 'J':
            j -= 1
            cntj += 1
        if cntj >= cnto and cnti >= cnto:
            ans = max(ans, min(cntj, cnto, cnti))
        i = nxt
    else:
        i += 1
print(ans)
