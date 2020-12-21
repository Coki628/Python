# -*- coding: utf-8 -*-

"""
stack
"""

from collections import deque

aN = list(map(str, input().split()))

stack = deque()
for i in range(len(aN)):
    if aN[i] in ('+', '-', '*'):
        b = stack.pop()
        a = stack.pop()
        stack.append(str(eval(a + aN[i] + b)))
    else:
        stack.append(aN[i])
print(stack.pop())
