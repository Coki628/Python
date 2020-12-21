# -*- coding: utf-8 -*-

"""
※すげー前に書いたやつ。
"""
 
N, Z, W = map(int, input().split())
card_list = list(map(int, input().split()))
 
X = Z
Y = W
 
# どちらのターンかを判別するフラグ
Xs_turn = True
 
# カードの枚数ループ
for i in range(N):
    # Xのターン
    if Xs_turn:
        # 最後の1枚の場合を考慮する
        if X < card_list[i] or i == N - 1:
            X = card_list[i]
            Xs_turn = False
        else:
            X = card_list[i]
    # Yのターン
    else:
        if Y > card_list[i] or i == N - 1:
            Y = card_list[i]
            Xs_turn = True
        else:
            Y = card_list[i]
print(abs(X - Y))