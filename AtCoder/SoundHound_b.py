# -*- coding: utf-8 -*-
 
S = input()
w = int(input())

ans = ""
for i in range(len(S)):
    if i == 0:
        ans += S[i]
        continue
    if i % w == 0:
        ans += S[i]
print(ans)