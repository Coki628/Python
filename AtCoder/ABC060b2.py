# -*- coding: utf-8 -*-

A, B, C = map(int, input().split())

# 確認するのは1からBまででいい(Bの倍数分は割り切れるから余りに影響を与えない)
for i in range(1, B+1):
    if (A * i) % B == C:
        print("YES")
        exit()
print("NO")