# -*- coding: utf-8 -*-
 
"""
ゲーム木の探索(ミニマックス法)
・自分のターンに得点を最大化、相手のターンに最小化
・全局面をdfsで回ってお互いが最善手を打った時の結果を割り出せる
"""
 
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
        return mn
 
# 再帰処理を0ターンからスタート
taka = dfs(0)
naoko = sm - taka
 
print(taka)
print(naoko)