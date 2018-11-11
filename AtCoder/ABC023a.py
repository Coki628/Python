# -*- coding: utf-8 -*-

X = int(input())

# print(X//10 + X%10)

ans = 0
while X != 0:
    ans += X % 10
    X //= 10
print(ans)