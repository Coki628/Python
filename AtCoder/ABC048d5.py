# -*- coding: utf-8 -*-

"""
・調査用
・ゲーム木の探索、後退解析、メモ化再帰
・まあ結局手掛かりはつかめなかったんだけど。
・で、答え見て、この場合は終了状態がどうなってるか注目する必要があったから、
　それを出力する部分を追加した。
・文字列の1文字変更にO(N)かからないようにしてみた版。
　まあ、とりあえず小さいケース見る限りは速くなってないけど。
"""

import sys
from collections import defaultdict, deque
from itertools import product

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

memo = defaultdict(str)
def rec(S):
    if memo[S] != '':
        return memo[S]
    N = len(S)
    res = True    
    end = True
    S = list(S)
    S1 = list(S[0])
    S2 = deque(S[1:])
    for i in range(N-2):
        cur = S2.popleft()
        if S1[-1] != S2[0]:
            end = False
            res &= rec(''.join(S1+list(S2)))
        S1.append(cur)
    # 終了時の盤面を出力
    if end:
        print('end: {0}'.format(S))
     # 1つも負けが返らなければここは負け
    if res:
        memo[''.join(S)] = False
        return False
    else:
        memo[''.join(S)] = True
        return True

# 同じ文字が隣り合わないかチェック
def check(prod):
    for j in range(1, len(prod)):
        if prod[j-1] == prod[j]:
            return False
    return True

for i in range(3, 7):
    for prod in product('abc', repeat=i):
        if not check(prod): continue
        S = ''.join(prod)
        res = rec(S)
        print('S: {0}, {1}'.format(S, res))
