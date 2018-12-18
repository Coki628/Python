# -*- coding: utf-8 -*-

"""
部分点狙い
・アームを細かく割ってどこでも行けるようにする
"""

N = int(input())
xN, yN = [0] * N, [0] * N
xN[0], yN[0] = map(int, input().split())
od_ev = (abs(xN[0]) + abs(yN[0])) % 2
max_len = abs(xN[0]) + abs(yN[0])
for i in range(1, N):
    xN[i], yN[i] = map(int, input().split())
    # まず大前提として偶奇が合うかどうか
    if od_ev != (abs(xN[i]) + abs(yN[i])) % 2:
        print(-1)
        exit()
    # 最大距離の記録
    max_len = max(abs(xN[i]) + abs(yN[i]), max_len)

print(max_len)
print(*([1] * max_len))
for i in range(N):
    ans = ''
    # 必要な場所への移動
    if xN[i] < 0:
        ans += 'L' * abs(xN[i])
    elif xN[i] > 0:
        ans += 'R' * xN[i]
    if yN[i] < 0:
        ans += 'D' * abs(yN[i])
    elif yN[i] > 0:
        ans += 'U' * yN[i]
    # 余った長さを適当に動かす
    for i in range(max_len - len(ans)):
        if i % 2:
            ans += 'L'
        else:
            ans += 'R'
    print(ans)
