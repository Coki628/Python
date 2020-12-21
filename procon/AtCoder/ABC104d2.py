# -*- coding: utf-8 -*-

"""
参考：http://tutuz.hateblo.jp/entry/2018/08/06/074104
　　　https://www.hamayanhamayan.com/entry/2018/08/05/232127
・別解
・予め必要なA,C,?の個数は集めておく。
・Bの位置を決めて、4つの選び方パターンを全部数え上げる。
・3べきの計算はpowの第三引数modで高速化
"""

MOD = 10 ** 9 + 7
S = input()

# あるBの位置よりも左にあるAと?の個数
A, LQ = [0] * len(S), [0] * len(S)
for i in range(len(S)-1):
    a = lq = 0
    if S[i] == 'A':
        a += 1
    if S[i] == '?':
        lq += 1
    A[i+1] = A[i] + a
    LQ[i+1] = LQ[i] + lq
# あるBの位置よりも右にあるCと?の個数
C, RQ = [0] * len(S), [0] * len(S)
for i in range(len(S)-1, 0, -1):
    c = rq = 0
    if S[i] == 'C':
        c += 1
    if S[i] == '?':
        rq += 1
    C[i-1] = C[i] + c
    RQ[i-1] = RQ[i] + rq

ans = 0
for i in range(len(S)):
    if S[i] == '?' or S[i] == 'B':
        # (A,B,C) と選ぶ場合(後ろの3べきは使わなかった?の分)
        case1 = (A[i] * C[i] * pow(3, LQ[i] + RQ[i], MOD)) % MOD
        # (A,B,?) と選ぶ場合(3べきの指数が負になるとおかしくなるので下限0にする)
        case2 = (A[i] * RQ[i] * pow(3, max(LQ[i] + RQ[i] - 1, 0), MOD)) % MOD
        # (?,B,C) と選ぶ場合
        case3 = (LQ[i] * C[i] * pow(3, max(LQ[i] + RQ[i] - 1, 0), MOD)) % MOD
        # (?,B,?) と選ぶ場合
        case4 = (LQ[i] * RQ[i] * pow(3, max(LQ[i] + RQ[i] - 2, 0), MOD)) % MOD
        # 全部足し合わせてMOD
        ans = (ans + case1 + case2 + case3 + case4) % MOD

print(ans)
