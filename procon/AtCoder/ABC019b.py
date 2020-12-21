# -*- coding: utf-8 -*-

s = input()

c = s[0]
cnt = 1
ans = ''
for i in range(1, len(s)):
    if s[i] == s[i-1]:
        cnt += 1
    else:
        ans += s[i-1] + str(cnt)
        cnt = 1
    if i == len(s)-1:
        ans += s[i] + str(cnt)
print(ans)
