# -*- coding: utf-8 -*-

N, K = map(int, input().split())
aN = [int(input()) for i in range(N)]

# 平均に均すか、極端にでかいやつに合わせる
print(max(-(-sum(aN) // K), max(aN)))
