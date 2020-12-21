# -*- coding: utf-8 -*-

N = int(input())
S = input()

# 初期値設定
cnt = S[1:].count("E")
ans = cnt
# 結局自分と1個前しか変化しない
for i in range(1, N):
    if S[i] == "E":
        cnt -= 1
    if S[i-1] == "W":
        cnt += 1
    ans = min(ans, cnt)

print(ans)