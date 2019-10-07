# -*- coding: utf-8 -*-

"""
・蟻本演習2-3-9、自力AC！けど危なかった。。
・ゲームDP、メモ化再帰
・最後の結果を出す時に遷移前に計算することで、相手側の情報を保持しなくてもいいようにした。
・前の解法とは違う書き方だったけど、結果的に自力で通せて良かった。
・速度もちょっと速くなってた。pythonTLE => 1.5秒AC、pypy0.8秒AC => 0.7秒AC。
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

N, _, W = MAP()
A = [W] + LIST()

# memo[i][j] := 現在iターンで、山札からj枚目まで取った状態
memo = list2d(2, N+1, None)
def rec(turn, cnt):
    if memo[turn][cnt] is not None:
        return memo[turn][cnt]
    # 先手番
    if turn % 2 == 0:
        res = -INF
        for nxt in range(cnt+1, N):
            res = max(res, rec(1-turn, nxt))
        # 最後まで取る時は、結果を取得
        res = max(res, abs(A[N]-A[cnt]))
    # 後手番
    else:
        res = INF
        for nxt in range(cnt+1, N):
            res = min(res, rec(1-turn, nxt))
        res = min(res, abs(A[cnt]-A[N]))
    memo[turn][cnt] = res
    return res

ans = rec(0, 0)
print(ans)
