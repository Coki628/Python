# -*- coding: utf-8 -*-

"""
アメを配置していく時に、行列の合計に加算してKと比べる
"""

R,C,K = map(int, input().split())
N = int(input())

# 行R*列Cの部屋(各位置のアメ獲得数を保持)
RC = [[0] * C for i in range(R)]
ans = 0
for i in range(N):
    r,c = map(int, input().split())
    # 列固定で各行に+1
    for i in range(R):
        if i != r-1:
            RC[i][c-1] += 1
            if RC[i][c-1] == K:
                ans += 1
            if RC[i][c-1] == K+1:
                ans -= 1
    # 行固定で各列に+1
    for i in range(C):
        if i != c-1:
            RC[r-1][i] += 1
            if RC[r-1][i] == K:
                ans += 1
            if RC[r-1][i] == K+1:
                ans -= 1
    # 重複対策
    RC[r-1][c-1] += 1
    if RC[r-1][c-1] == K:
        ans += 1
    if RC[r-1][c-1] == K+1:
        ans -= 1
    
print(ans)
