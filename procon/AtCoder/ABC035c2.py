# -*- coding: utf-8 -*-

# いもす法

N,Q = map(int, input().split())

# 各区間の始点と終点を集計していく
xN = [0] * N
for i in range(Q):
    l,r = map(int, input().split())
    # 区間の始点と終点を加減する
    xN[l-1] += 1
    # r == Nは終点なしと言える(out of rangeだし)
    if r != N:
        xN[r] -= 1

# 初期値
ans = str(xN[0] % 2)
# 累積和ループ
for i in range(1,N):
    xN[i] += xN[i-1]
    ans += str(xN[i] % 2)
    
print(ans)