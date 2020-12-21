# -*- coding: utf-8 -*-

# 尺取法

N = int(input())
aN = list(map(int, input().split()))

l,r,ans = 0,0,0

# 外ループで左を動かす
while l < N:
    # 内ループは条件を満たす限り右を動かす
    while r < N-1 and aN[r] < aN[r+1]:
        r += 1
    # 左右の差分+1
    ans += r - l + 1
    # 左が右に追いついたら、右も左に合わせて+1する
    if l == r:
        r += 1
    l += 1
print(ans)