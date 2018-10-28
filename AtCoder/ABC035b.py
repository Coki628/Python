# -*- coding: utf-8 -*-

S = input()
T = int(input())

drn = [0, 0]
q = S.count('?')

# とりあえず?以外の移動位置
for c in S:
    if c == 'L':
        drn[0] -= 1
    elif c == 'R':
        drn[0] += 1
    elif c == 'U':
        drn[1] += 1
    elif c == 'D':
        drn[1] -= 1
dst = abs(drn[0]) + abs(drn[1])

# ?の数だけ遠ざけるか近づけるか
if T == 1:
    print(dst + q)
elif T == 2:
    if dst - q >= 0:
        print(dst - q)
    else:
        # マイナスになる場合は偶奇で0か1
        print(abs(dst - q) % 2)