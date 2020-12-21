# -*- coding: utf-8 -*-

N,W = map(int, input().split())
wN = []
vN = []
for i in range(N):
    w,v = map(int, input().split())
    wN.append(w)
    vN.append(v)

dp = [[0] * (3*(N-1)+1) for i in range(N+1)]

# 答えが明らかな範囲はDPに含めない
if W < wN[0]:
    print(0)
    exit()
elif W >= sum(wN):
    print(sum(vN))
    exit()

# wNの範囲はかなり限定されているので、オフセットを用意
# (-1するのは0地点を残すため)
offset = wN[0]-1
W -= offset

for i in range(N):
    for j in range(offset, (3*(N-1)+1) + offset):

        if wN[i] <= j:
            # max(0,～)で0未満を探しに行くのを避ける
            dp[i+1][j-offset] = max(dp[i][max(0,j-wN[i]-offset)] + vN[i], dp[i][j-offset])
        else:
            dp[i+1][j-offset] = dp[i][j-offset]

print(dp[N][W])