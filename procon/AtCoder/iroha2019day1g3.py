# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/iroha2019-day1/editorial-G.pdf
・いわゆる普通のminmax系DP(2次元+内ループ)
・とりあえずこれはメモ化再帰実装
・使わない日は考慮しなくていいってのに気付くべきだった。
・このくらいのは自力で解きたかったなぁ。。
"""

import sys

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

N,M,K=MAP()
A=[0]+LIST()

# memo[i][j] := i日目にj回使った状態
memo=list2d(N+1, M+1, 0)
def rec(cur, m):
    # 終了条件
    if m==M:
        # M回終わって、あとK日で終わりまで行ければOK
        if cur+K>N:
            return 0
        else:
            return -INF
    if memo[cur][m]:
        return memo[cur][m]
    res=-INF
    # 今からK日の範囲内で全ての日に遷移してみる
    for i in range(cur+1, cur+1+K):
        if i>N: break
        # 戻ってきた時に今の分を足す
        res=max(res, rec(i, m+1)+A[i])
    memo[cur][m]=res
    return res

res=rec(0, 0)
if res!=-INF:
    print(res)
else:
    print(-1)
