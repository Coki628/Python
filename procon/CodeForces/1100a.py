# -*- coding: utf-8 -*-

N, K = map(int, input().split())
tabs = list(map(int, input().split()))

ans = 0
for i in range(K):
    s = 0
    e = 0
    for j in range(N):
        if j % K != i:
            if tabs[j] == 1:
                s += 1
            elif tabs[j] == -1:
                e += 1
    ans = max(abs(e - s), ans)
print(ans)
