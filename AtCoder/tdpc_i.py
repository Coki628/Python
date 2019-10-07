# -*- coding: utf-8 -*-

"""
参考：http://fluffyowl.hatenablog.com/entry/2017/09/26/210340
　　　http://algoogle.hadrori.jp/tdpc/i/
・蟻本演習2-3-10
・区間DP、メモ化再帰では何度かやったことあったけど、普通のDPでの区間DPは初めてやった。
・3重ループの回し方に目から鱗。なるほど、確かにこれなら狭い区間から順番に確認できる。
・意外と連鎖の起こる状況は限定されているので、if文2つで条件作った。
・計算量は300^3=2700万、pythonTLEでpypyAC0.3秒。
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

S = input()
N = len(S)

# dp[l][r] := 区間[l,r)について、iwiを消せる最大回数
dp = list2d(N+1, N+1, 0)
# 初期化：3文字で最初からでiwiが成立している所を確認
for l in range(N-3):
    r = l + 3
    if S[l:r] == 'iwi':
        dp[l][r] = 1
# [l,r)の長さループ
for d in range(4, N+1):
    # 左端ループ
    for l in range(N+1-d):
        r = l + d
        # [l,m)と[m,r)の全てから最適な値を集めてくる
        for m in range(l+1, r):
            dp[l][r] = max(dp[l][r], dp[l][m]+dp[m][r])
        # 連鎖の可能性がある場合
        if (r-l) % 3 == 0 and S[l] == S[r-1] == 'i':
            # i***(全消し)***wiが成立するなら+1
            if (r-l)//3-1 == dp[l+1][r-2] and S[r-2] == 'w':
                dp[l][r] = max(dp[l][r], dp[l+1][r-2]+1)
            # iw***(全消し)***iが成立するなら+1
            if (r-l)//3-1 == dp[l+2][r-1] and S[l+1] == 'w':
                dp[l][r] = max(dp[l][r], dp[l+2][r-1]+1)
print(dp[0][N])
