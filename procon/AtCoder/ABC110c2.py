# -*- coding: utf-8 -*-

S = input()
T = input()

# 1対1の変換表(dict)
start = {}
goal = {}

for i in range(len(S)):
    s = S[i]
    t = T[i]
    # 変換表(変換前S)の該当箇所がまだ空いてるかどうか
    if s in start:
        # 埋まっている場合、今回の組み合わせと一致しているかどうか
        if start[s] != t:
            print('No')
            exit()
    # 空いていれば変換表を埋める
    else:
        start[s] = t
    # 変換後Tも同じようにやる
    if t in goal:
        if goal[t] != s:
            print('No')
            exit()
    else:
        goal[t] = s

print('Yes')