# -*- coding: utf-8 -*-

"""
人の参考にほぼ写経して、動かしてみて噛み砕いた版
"""

import copy

N, K = map(int, input().split())
S = input()

C = {}
# dictを全アルファベット0で初期化
for c in range(ord('a'), ord('z')+1):
    C[chr(c)] = 0
# 各アルファベットが元のSにそれぞれいくつ含まれていたか
for c in S:
    C[c] += 1

def okok(ind, num):
    # 数を減らしながら調べるのを何回もやるのでコピーして使う
    C2 = copy.deepcopy(C)
    # 確定したい場所より後ろでループ
    for i in range(ind, N):
        # ある文字が、元の文字列の同じ範囲で使われているか
        if C2[S[i]] > 0:
            # 集計結果から減らしておく
            C2[S[i]] -= 1
        else:
            # 一致しない文字は移動対象になるので、Kが減る
            num -= 1
    # 最終的にKが足りたか判定
    return num >= 0

T = ''
for i in range(N):
    # a～zのループ
    for j in range(ord('a'), ord('z')+1):
        c = chr(j)
        # ある文字が使われていたか
        if C[c] > 0:
            C[c] -= 1
            # 確定したい場所の文字と一致していなければ、移動可能数を減らす
            if S[i] != c:
                K -= 1
            # 残りの文字列を整えるのにKで足りるかの判定
            if okok(i+1, K):
                # 大丈夫ならこの文字を追加
                T += c
                # 確定したい場所を1つ進めるので内ループ抜ける
                break
            # 結局この文字は使わなかったのでKのカウントは戻す
            if S[i] != c:
                K += 1
            # 元のSに含まれていた数も戻す
            C[c] += 1
print(T)
