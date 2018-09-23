# -*- coding: utf-8 -*-

S = input()
T = input()

# 1対1の対応表
start = [-1] * 26
goal = [-1] * 26

for i in range(len(S)):
    # アルファベットを0～25の数値に変換
    a = ord(S[i]) - ord('a')
    b = ord(T[i]) - ord('a')
    # 対応表の該当箇所がまだ空いてるかどうか
    if start[a] != -1 or goal[b] != -1:
        # 埋まっている場合、今回の組み合わせと一致しているかどうか
        if start[a] != b or goal[b] != a:
            print('No')
            exit()
    # 空いていれば対応表を埋める
    else:
        start[a] = b
        goal[b] = a

print('Yes')