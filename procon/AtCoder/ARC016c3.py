# -*- coding: utf-8 -*-

"""
参考：https://www.creativ.xyz/arc016c-605/
　　　https://tokumini.hatenablog.com/entry/2018/08/23/112450
　　　https://kimiyuki.net/blog/2016/04/28/dice-and-expected-value/
　　　https://math314.hateblo.jp/entry/2013/12/12/232045
・蟻本演習3-4-2
・bitDP、期待値DP、メモ化再帰
・練習のために、メモ化再帰版も実装。
・集合の持ち方が逆。でもこっちのが直感的かなとも思う。
・制約余裕だからか、再帰でも速度落ちなくてpythonAC0.04秒、pypyAC0.2秒。
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

N, M = MAP()

P = list2d(M, N, 0)
C = [0] * M
for i in range(M):
    m, cost = MAP()
    C[i] = cost
    for j in range(m):
        a, p = MAP()
        a -= 1
        P[i][a] = p / 100

# memo[S] := 既に出た面の集合Sの状態で、それ以降に振る回数の期待値の最小値
memo = [-1] * (1<<N)
def rec(S):
    if memo[S] != -1:
        return memo[S]
    # 全部の面が出たらその先に振る回数は0
    if S == (1<<N)-1:
        return 0
    res = INF
    # 各くじについて試して、今の集合の状態になれる最小を取る
    for i in range(M):
        tmp = 0
        for j in range(N):
            if not S & (1<<j):
                tmp += P[i][j]
        if tmp == 0:
            continue
        # まだ出ていない面が出るまでに振る回数 * コスト
        nxt = 1 / tmp * C[i]
        # 今の集合に含まれない部分について再帰的に調べていく
        for j in range(N):
            if not S & (1<<j):
                # それ以降に振る必要のある回数(コスト織込済)
                nxt += P[i][j] / tmp * rec(S|(1<<j))
        res = min(res, nxt)
    memo[S] = res
    return res

print(rec(0))
