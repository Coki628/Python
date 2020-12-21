# -*- coding: utf-8 -*-

# 入力を各変数に格納
n, m = map(int, input().split())

cells = [[1 for i in range(n)] for j in range(m)]

# カードを裏返す関数
def turnCard(_y, _x):

    global cells

    if cells[_y][_x] == 0:
        cells[_y][_x] = 1
    elif cells[_y][_x] == 1:
        cells[_y][_x] = 0

# 各マスを周回して8方向に裏返し処理
for y in range(m):
    for x in range(n):

        # そのマスを裏返す
        turnCard(y, x)
        # 左端以外
        if x != 0:
            # 左隣を裏返す
            turnCard(y, x-1)
        # 右端以外
        if x != n-1:
            # 右隣を裏返す
            turnCard(y, x+1)
        # 最上段以外
        if y != 0:
            # ひとつ上を裏返す
            turnCard(y-1, x)
        # 最下段以外
        if y != m-1:
            # ひとつ下を裏返す
            turnCard(y+1, x)
        # 左端と最上段以外
        if x != 0 and y != 0:
            # 左斜め上を裏返す
            turnCard(y-1, x-1)
        # 右端と最上段以外
        if x != n-1 and y != 0:
            # 右斜め上を裏返す
            turnCard(y-1, x+1)
        # 左端と最下段以外
        if x != 0 and y != m-1:
            # 左斜め下を裏返す
            turnCard(y+1, x-1)
        # 右端と最下段以外
        if x != n-1 and y != m-1:
            # 右斜め下を裏返す
            turnCard(y+1, x+1)
        
# 各マスを周回して裏のマスで+1
ans = 0
for y in range(m):
    for x in range(n):
        if cells[y][x] == 0:
            ans += 1
print(ans)