# -*- coding: utf-8 -*-

H,W = map(int, input().split())
HW = []
for i in range(H):
    HW.append(list(map(int, input().split())))

ans = []
for i in range(H):
    for j in range(W):
        # 該当マスが奇数
        if HW[i][j] % 2 != 0:
            if i != 0:
                if HW[i-1][j] % 2 != 0:
                    HW[i-1][j] += 1
                    HW[i][j] -= 1
                    ans.append(str(i+1) + ' ' + str(j+1) + ' ' + str(i-1+1) + ' ' + str(j+1))
                    continue
            if j != 0:
                if HW[i][j-1] % 2 != 0:
                    HW[i][j-1] += 1
                    HW[i][j] -= 1
                    ans.append(str(i+1) + ' ' + str(j+1) + ' ' + str(i+1) + ' ' + str(j-1+1))
                    continue
            if i != H-1:
                if HW[i+1][j] % 2 != 0:
                    HW[i+1][j] += 1
                    HW[i][j] -= 1
                    ans.append(str(i+1) + ' ' + str(j+1) + ' ' + str(i+1+1) + ' ' + str(j+1))
                    continue
            # 最終的に周り全部偶数でも右隣りに移動
            if j != W-1:
                HW[i][j+1] += 1
                HW[i][j] -= 1
                ans.append(str(i+1) + ' ' + str(j+1) + ' ' + str(i+1) + ' ' + str(j+1+1))

# 操作回数から出力
print(len(ans))
for s in ans:
    print(s)