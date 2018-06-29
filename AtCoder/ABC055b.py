# -*- coding: utf-8 -*-

N = int(input())

ans = 1
for i in range(N):
    ans = ans * (i + 1)

# 大き過ぎる値を調整する
ans = ans % (10 ** 9 + 7) 

print(ans)