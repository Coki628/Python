# -*- coding: utf-8 -*-

"""
・pypyでもTLEなのでC++でAC0.2秒。
・ゲームDP、メモ化再帰
・円状なのでNでmod取って周期にする以外は割と普通。
・こういうl,rの区間動かしてくタイプのゲームDPはEDPC-LでもpypyTLEしたので、まあしょうがないか。
・計算量自体は2000のNだから400万(ターン偶奇考えても800万)程度なんだけどね。再帰弱いからね。
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

N = INT()
A = LIST(N)

# ゲームDP、メモ化再帰
memo = list3d(N, N, 2, -1)
def rec(l, r, turn):
    if turn == N:
        return 0
    if memo[l][r][turn&1] != -1:
        return memo[l][r][turn&1]
    # 自分の手番は両方試す
    if turn & 1 == 0:
        res = max(rec((l-1)%N, r, turn+1) + A[(l-1)%N], rec(l, (r+1)%N, turn+1) + A[(r+1)%N])
    # 相手の手番は必ず大きい方を取る
    else:
        if A[(l-1)%N] >= A[(r+1)%N]:
            res = rec((l-1)%N, r, turn+1)
        else:
            res = rec(l, (r+1)%N, turn+1)
    memo[l][r][turn&1] = res
    return res

ans = 0
for i in range(N):
    # 最初の1つを全箇所試す
    ans = max(ans, rec(i, i, 1) + A[i])
print(ans)
