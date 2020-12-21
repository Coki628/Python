# -*- coding: utf-8 -*-

"""
参考：submission: 48771839
・横に揃えば絶対消えるんだから、こうすれば良かった…。
"""

S = input()

grid = [[' '] * 4 for i in range(4)]

# 横ブロックは1行目だけ使う
def hori():
    if ''.join(grid[0]) == '**  ':
        grid[0][2] = '*'
        grid[0][3] = '*'
        print(0+1, 2+1)
    else:
        grid[0][0] = '*'
        grid[0][1] = '*'
        print(0+1, 0+1)

# 縦ブロックは2,3行目だけ使う
def vert():
    for i in range(4):
        if grid[1][i] == ' ':
            grid[1][i] = '*'
            grid[2][i] = '*'
            print(1+1, i+1)
            return

for s in S:
    if s == '1':
        hori()
    if s == '0':
        vert()
    # check
    for i in range(4):
        if ''.join(grid[i]) == '****':
            for j in range(4):
                grid[i][j] = ' '
