# -*- coding: utf-8 -*-

S = input()

grid = [[' '] * 4 for i in range(4)]

def hori():
    for i in range(4):
        if ''.join(grid[i]) == '**  ':
            grid[i][2] = '*'
            grid[i][3] = '*'
            print(i+1, 2+1)
            return
        elif ''.join(grid[i]) == '*  *':
            grid[i][1] = '*'
            grid[i][2] = '*'
            print(i+1, 1+1)
            return
        elif ''.join(grid[i]) == '  **':
            grid[i][0] = '*'
            grid[i][1] = '*'
            print(i+1, 0+1)
            return
    for i in range(4):
        for j in range(3):
            if grid[i][j] == ' ' and grid[i][j+1] == ' ':
                grid[i][j] = '*'
                grid[i][j+1] = '*'
                print(i+1, j+1)
                return

def vert():
    for i in range(3):
        if ''.join(grid[i]) == '*** ' and ''.join(grid[i+1]) == '*** ':
            grid[i][3] = '*'
            grid[i+1][3] = '*'
            print(i+1, 3+1)
            return
        elif ''.join(grid[i]) == '** *' and ''.join(grid[i+1]) == '** *':
            grid[i][2] = '*'
            grid[i+1][2] = '*'
            print(i+1, 2+1)
            return
        elif ''.join(grid[i]) == '* **' and ''.join(grid[i+1]) == '* **':
            grid[i][1] = '*'
            grid[i+1][1] = '*'
            print(i+1, 1+1)
            return
        elif ''.join(grid[i]) == ' ***' and ''.join(grid[i+1]) == ' ***':
            grid[i][0] = '*'
            grid[i+1][0] = '*'
            print(i+1, 0+1)
            return
    for i in range(3):
        for j in range(4):
            if grid[i][j] == ' ' and grid[i+1][j] == ' ':
                grid[i][j] = '*'
                grid[i+1][j] = '*'
                print(i+1, j+1)
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
