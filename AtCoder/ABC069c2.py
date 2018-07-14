# -*- coding: utf-8 -*-

N = int(input())
aN = list(map(int, input().split()))

# okは4の倍数
ok = 0
# ngは4の倍数でもexcpでもない数
ng = 0
# excpはそのグループ内のペアに限って4の倍数になれる数(→ようは4の倍数以外の偶数だった)
excp = 0
for i in range(N):
    if aN[i] % 4 == 0:
        ok += 1
    elif (aN[i] + 2) % 4 == 0:
        excp += 1
    else:
        ng += 1

# excpは全部1箇所にまとめてng1つ扱い
if excp > 0:
    ng += 1

# 外をngで囲まれてもいいので、okが1個少なくてもYes
if ok + 1 >= ng:
    print("Yes")
else:
    print("No")