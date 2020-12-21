# -*- coding: utf-8 -*-
 
N = int(input())
aN = list(map(int, input().split()))

ans = 0
for i in range(N):
    sm = xor = 0
    for j in range(i, N):
        sm += aN[j]
        xor ^= aN[j]
        if sm == xor:
            ans += 1
        # 一度合わなくなったら、そこから先で合うことはないはず
        else:
            break
print(ans)
