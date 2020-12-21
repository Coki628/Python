# -*- coding: utf-8 -*-

"""
参考：https://suikaba.hatenablog.com/entry/2017/08/27/181249
・蟻本演習3-4-3
・bitDP、スライドbitDP
・3次元DPの2,3次元目で、2つの集合の情報を持つ。
・計算量は一番深いループが510*2^7*2^8*2*8=約2億6700万、、、
　だけど、最後の2*8は遷移させない時はつかないから、510*2^7*2^8だけだと約1700万。
・全部回ってもpypyAC1.7秒で、0からは遷移させないように枝刈りしたら0.5秒まで速くなった。
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

N, L = MAP()
# 文字列を長さとビットの情報で持つ
word = list2d(8, 1<<8, 0)
for i in range(N):
    S = input()
    bit = 0
    for i, s in enumerate(S):
        if s == '1':
            bit |= 1<<i
    word[len(S)-1][bit] = 1

# dp[i][S1][S2] := i文字目まで見て、直前7文字の状態がS1、直前8箇所の区切りの有無の状態がS2の時に作れる文字列の総数
dp = list3d(L+1, 1<<7, 1<<8, 0)
dp[0][0][1] = 1
msk1 = (1<<7) - 1
msk2 = (1<<8) - 1
for i in range(L):
    # 直前7文字の状態
    for S1 in range(1<<7):
        # 直前8箇所の区切りの有無の状態
        for S2 in range(1<<8):
            # 0からは遷移させる意味ない
            if not dp[i][S1][S2]:
                continue
            for j in range(2):
                # 今回の文字0or1を合わせた、直前8文字の状態
                S3 = S1<<1 | j
                flag = False
                for k in range(8):
                    # k文字前より手前を削るマスク
                    msk3 = (1<<(k+1)) - 1
                    # k文字前に区切りがある and 直前k文字を使った単語がある
                    if S2 & (1<<k) and word[k][S3&msk3]:
                        flag = True
                dp[i+1][S3&msk1][(S2<<1&msk2)|flag] += dp[i][S1][S2]
                dp[i+1][S3&msk1][(S2<<1&msk2)|flag] %= MOD

ans = 0
for S1 in range(1<<7):
    for S2 in range(1<<8):
        # 最後が区切りになっていればOK
        if S2 & 1:
            ans += dp[L][S1][S2]
            ans %= MOD
print(ans)
