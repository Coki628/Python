# -*- coding: utf-8 -*-

H, W = map(int, input().split())
HW = []
for i in range(H):
    HW.append(list(input()))

# 8方向を見に行く用の配列を準備しておく
dx = [1, 0, -1, 0, 1, -1, -1, 1]
dy = [0, 1, 0, -1, 1, 1, -1, -1]

# 各マスを周回
for i in range(H):
    for j in range(W):
        # 今いるマスが#なら何もしない
        if HW[i][j] == "#":
            continue
        # 8方向について#を数えていく
        cnt = 0
        for k in range(8):
            # 各方向にずらした位置を保持
            ni = i + dy[k]
            nj = j + dx[k]
            # index範囲外の時も何もしない
            if ni < 0 or ni >= H:
                continue
            if nj < 0 or nj >= W:
                continue
            
            if HW[ni][nj] == "#":
                cnt += 1
        HW[i][j] = cnt
# 1行ずつ文字列に変換して出力
for i in range(H):
    print("".join(map(str, (HW[i]))))