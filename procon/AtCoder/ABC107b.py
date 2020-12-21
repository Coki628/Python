# -*- coding: utf-8 -*-

H,W = map(int, input().split())
HW = []
for i in range(H):
    instr = input()
    if instr.find('#') != -1:
        HW.append(list(instr))
    else:
        H -= 1

# 行列の入れ替え
HW2 = []
for x in zip(*HW):
    HW2.append(x)

HW3 = []
for i in range(W):
    instr = ''.join(HW2[i])
    if instr.find('#') != -1:
        HW3.append(HW2[i])

# もう1回入れ替えて戻す
HW4 = []
for x in zip(*HW3):
    HW4.append(x)

for i in range(H):
    print(''.join(HW4[i]))