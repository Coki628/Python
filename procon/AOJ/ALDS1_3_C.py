# -*- coding: utf-8 -*-

"""
連結リスト
"""

from collections import deque

N = int(input())

que = deque()
for i in range(N):
    instr =  input()
    if instr == 'deleteFirst':
        que.popleft()
    elif instr == 'deleteLast':
        que.pop()
    else:
        # 計算はしないのでstrのままでいい(しかもこの方が速い)
        com, num = instr.split()
        if com == 'insert':
            que.appendleft(num)
        elif com == 'delete':
            try:
                que.remove(num)
            except ValueError:
                pass
print(*que)
