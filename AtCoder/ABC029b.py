# -*- coding: utf-8 -*-

ans = 0
for i in range(12):
    s = input()
    if s.find('r') != -1:
        ans += 1
print(ans)