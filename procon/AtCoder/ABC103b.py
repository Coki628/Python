# -*- coding: utf-8 -*-

S = list(input())
T = list(input())

for s in range(len(S)):
    if str(S) == str(T):
        print("Yes")
        exit()
    tmp = S.pop(0)
    S.append(tmp)
print("No")