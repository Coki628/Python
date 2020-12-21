# -*- coding: utf-8 -*-

H,W = map(int, input().split())
HW = []
for i in range(H):
    HW.append(list(map(int, input().split())))

ans = []
for i in range(H):
    for j in range(W):
        # 該当マスが奇数の時に処理
        if HW[i][j] % 2 == 1:
            # 最終行以外全部下に移動
            if i != H-1:
                HW[i+1][j] += 1
                HW[i][j] -= 1
                ans.append([i+1, j+1, i+1+1, j+1])
            else:
                # 最終行は最後のマス以外全部右に移動
                if j != W-1:
                    HW[i][j+1] += 1
                    HW[i][j] -= 1
                    ans.append([i+1, j+1, i+1, j+1+1])

# 操作回数から出力
print(len(ans))
for s in ans:
    print(' '.join(map(str, s)))