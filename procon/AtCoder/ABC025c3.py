# -*- coding: utf-8 -*-

"""
ゲーム木の探索(ミニマックス法) - メモ化再帰で高速化版
・一度その盤面状況での得点が分かったら、他のルートからそこに来ても未来は変わらない。
・つまり、全マス埋まってなくても、それ以上潜る必要がない。
・各盤面状況での得点を保持する。
"""

import numpy as np

# 入力受け取りついでに全得点の合計も取っておく
sm = 0
b = [None] * 2
for i in range(2):
    b[i] = list(map(int, input().split()))
    sm += sum(b[i])
c = [None] * 3
for i in range(3):
    c[i] = list(map(int, input().split()))
    sm += sum(c[i])

# 盤面の状態を保持する
field = [['*'] * 3 for i in range(3)]
field = np.array((field))
# 盤面状況：得点 のペアを保持する
memo = {}

# その時の盤面でのtakaの得点を算出
def calc():
    taka = 0
    for i in range(2):
        for j in range(3):
            if field[i][j] == field[i+1][j]:
                taka += b[i][j]
    for i in range(3):
        for j in range(2):
            if field[i][j] == field[i][j+1]:
                taka += c[i][j]
    return taka

def dfs(turn):
    # 盤面が埋まったら
    if turn == 9:
        # takaの得点を算出して前のターンに戻る
        return calc()
    # メモを確認して、一致があれば得点を返す(これ以上再帰処理しない)
    if ''.join(field.ravel()) in memo:
        return memo[''.join(field.ravel())]
    # takaのターン
    if turn % 2 == 0:
        mx = -1
        for i in range(3):
            for j in range(3):
                if field[i][j] == '*':
                    # 空いている任意のマスにo
                    field[i][j] = 'o'
                    # 再帰呼び出しで次のターンに入る
                    mx = max(dfs(turn + 1), mx)
                    # 再帰処理が返ってきたので、そのマスを戻す
                    field[i][j] = '*'
        # 盤面状況：得点 をメモする
        memo[''.join(field.ravel())] = mx
        # 全部のマスを回りきったら、盤面埋まってなくても前のターンに戻る
        return mx
    # naokoのターン(大小逆でほぼ同じ)
    else:
        mn = float('inf')
        for i in range(3):
            for j in range(3):
                if field[i][j] == '*':
                    field[i][j] = 'x'
                    mn = min(dfs(turn + 1), mn)
                    field[i][j] = '*'
        memo[''.join(field.ravel())] = mn
        return mn

# 再帰処理を0ターンからスタート
taka = dfs(0)
naoko = sm - taka

print(taka)
print(naoko)