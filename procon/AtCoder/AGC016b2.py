# -*- coding: utf-8 -*-

"""
・700点自力AC
・実験から、あり得るパターンの規則性が出たので、それを列挙して比較。
・公式解も読み解こうとしたけど、超考察ゲーでうまくできなかった。。
　写経だけならできるだろうけど、自分で導ける気が全くしない。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST()

li = [{1: N}]
for i in range(1, N-1):
    # あり得るパターン(iがa個以下でi+1がb個以下)を列挙
    li.append({i: i, i+1: min(N-i+max(N-i-2, 0), N)})
li.append({N-1: N})

C = Counter(A)
for d in li:
    cnt = 0
    for k, v in d.items():
        if C[k] <= v:
            # 集計済をカウント
            cnt += C[k]
        else:
            break
    else:
        # 条件の中で全要素集計できたらOK
        if cnt == N:
            Yes()
            exit()
No()
