# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

ans = 1
add = 0
for i in range(1,N):
    # ひとつ前の値と比べる
    if aN[i] > aN[i-1]:
        # 大きい値が続く限り追加分が増えていく
        add += 1
        ans += 1 + add
    # 小さかったら追加分は0に戻す
    else:
        ans += 1
        add = 0
print(ans)