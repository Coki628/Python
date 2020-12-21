# -*- coding: utf-8 -*-

H,W = map(int, input().split())

# 1マス分の必要な情報を保持する
class Cell:
    def __init__(self, h, w, val):
        self.h = h
        self.w = w
        self.val = val

# 全マスを一列にする
Cells = []
for i in range(H):
    tmp = list(map(int, input().split()))
    for j in range(W):
        Cells.append(Cell(i+1, j+1, tmp[j]))

ans = []
for i in range(len(Cells)-1):
    # 奇数マスは次のマスに順次値を移していく
    if Cells[i].val % 2 != 0:
        Cells[i].val -= 1
        Cells[i+1].val += 1
        ans.append(str(Cells[i].h) + ' ' + str(Cells[i].w) + ' ' + str(Cells[i+1].h) + ' ' + str(Cells[i+1].w))

# 操作回数から出力
print(len(ans))
for s in ans:
    print(s)