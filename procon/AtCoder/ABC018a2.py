# -*- coding: utf-8 -*-

aN = [int(input()) for i in range(3)]

bN = sorted(aN, reverse=True)

for i in range(len(aN)):
    print(bN.index(aN[i]) + 1)
