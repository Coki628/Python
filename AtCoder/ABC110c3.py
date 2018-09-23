# -*- coding: utf-8 -*-

S = input()
T = input()

# 1対1の変換表(dict)
start = {}
goal = {}

for i in range(len(S)):
    s = S[i]
    t = T[i]
    # 変換表の該当箇所がまだ空いてるかどうか
    if s in start or t in goal:
        # 埋まっている場合、今回の組み合わせと一致しているかどうか
        if start.get(s) != t or goal.get(t) != s:
            print('No')
            exit()
    # 空いていれば変換表を埋める
    else:
        start[s] = t
        goal[t] = s

print('Yes')