# -*- coding: utf-8 -*-

grid = [list(map(int, input().split())) for i in range(3)]

# 条件1個少なくてもいけた
if grid[0][1] + grid[1][0] - grid[1][1] == grid[0][0]\
        and grid[0][1] + grid[1][2] - grid[1][1] == grid[0][2]\
        and grid[1][0] + grid[2][1] - grid[1][1] == grid[2][0]\
        and grid[1][2] + grid[2][1] - grid[1][1] == grid[2][2]:
    print("Yes")
else:
    print("No")