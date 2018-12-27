# -*- coding: utf-8 -*-
 
N, C = map(int, input().split())
D = [list(map(int, input().split())) for i in range(C)]
cN = [list(map(int, input().split())) for i in range(N)]

# [mod 3でどれになるか][今の色] = 該当する数
memo = [[0] * C for i in range(3)]
for i in range(N):
    for j in range(N):
        memo[(i+j+2)%3][cN[i][j]-1] += 1

ans = float('inf')
# 変わる色を全組み合わせループ
for i in range(C):
    for j in range(C):
        for k in range(C):
            if i == j or j == k or k == i:
                continue
            sm = 0
            # 元の色ループ
            for l in range(C):
                # メモを元に、各塗り替えでどのくらい違和感が出るか算出
                sm += D[l][i] * memo[0][l]
                sm += D[l][j] * memo[1][l]
                sm += D[l][k] * memo[2][l]
            ans = min(sm, ans)
print(ans)
