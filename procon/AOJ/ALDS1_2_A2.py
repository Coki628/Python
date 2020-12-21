# -*- coding: utf-8 -*-

"""
bubble sort
"""

N = int(input())
aN = list(map(str, input().split()))

done = False
cnt = 0
start = 0
while not done:
    done = True
    # 始点を保持しておけば、内ループは短くできる(毎回先頭まで行く必要ない)
    for i in range(N-1, start, -1):
        if int(aN[i]) < int(aN[i-1]):
            tmp = aN[i]
            aN[i] = aN[i-1]
            aN[i-1] = tmp
            done = False
            cnt += 1
    start += 1
print(' '.join(aN))
print(cnt)
