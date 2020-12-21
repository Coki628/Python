# -*- coding: utf-8 -*-

"""
参考：https://tokumini.hatenablog.com/entry/2018/08/23/112450
　　　https://kimiyuki.net/blog/2016/04/28/dice-and-expected-value/
　　　https://math314.hateblo.jp/entry/2013/12/12/232045
・蟻本演習3-4-2
・bitDP、期待値DP
・再帰で回す人も多いっぽかったけど、前からDPしてるコードは
　自分が作ろうとしていた形に近かったので、ひとまずそれを参考に作成。
・基本的には、くじ1つだった場合に使えるDPに、各くじのループを挟むだけ。
・あと最小を取りに行く形になるので、初期化をINFにして、それに合わせて微調整。
・N<=10のbitDPなのでさすがに速度は余裕。pythonAC0.05秒、pypyAC0.2秒。
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

# dp[S] := まだ出ていない面の集合Sの状態から始めて、残る全ての面が出るまでに振る回数の期待値の最小値
dp = [INF] * (1<<N)
dp[0] = 0
for bit in range(1, 1<<N):
    # 各くじについて試して、今の集合の状態になれる最小を取る
    for i in range(M):
        tmp = 0
        for j in range(N):
            if bit & (1<<j):
                tmp += P[i][j]
        if tmp == 0:
            continue
        # まだ出ていない面が出るまでに振る回数 * コスト
        nxt = 1 / tmp * C[i]
        # 部分集合から必要な値を持ってきて足す
        for j in range(N):
            if bit & (1<<j):
                # それ以降に振る必要のある回数(コスト織込済)
                nxt += P[i][j] / tmp * dp[bit^(1<<j)]
        dp[bit] = min(dp[bit], nxt)
print(dp[-1])
