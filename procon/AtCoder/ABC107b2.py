# -*- coding: utf-8 -*-

H,W = map(int, input().split())
HW = []
for i in range(H):
    instr = input()
    # 残す行を選別
    if instr.find('#') != -1:
        HW.append(list(instr))
    # 残さない時は行数を-1
    else:
        H -= 1

# 残す列の情報を保持する
col = [False] * W
for i in range(H):
    for j in range(W):
        if HW[i][j] == '#':
            # 1回でも#があればその列はTrueになる
            col[j] = True

for i in range(H):
    for j in range(W):
        if col[j]:
            # 1文字ずつ出力
            print(HW[i][j], end='')
    # 行末の改行用
    print()