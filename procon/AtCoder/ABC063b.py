# -*- coding: utf-8 -*-

S = list(input())

S2 = list(set(S))

if len(S) == len(S2):
    print("yes")
else:
    print("no")