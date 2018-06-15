# -*- coding: utf-8 -*-

X, Y = map(int, input().split())

ans = 0
# Xを始点にしてYを超えるまで2倍していく
while X <= Y:
    ans += 1
    X *= 2

print(ans)