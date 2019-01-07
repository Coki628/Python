# -*- coding: utf-8 -*-

import sys
def input(): return sys.stdin.readline().strip()

from collections import deque, Counter

N = int(input())
aN = [input() for i in range(N)]

# 単体でペアになる括弧を除く
for i in range(N):
    stack = deque()
    for j in range(len(aN[i])):
        if aN[i][j] == '(':
            stack.append('(')
        else:
            if len(stack) != 0:
                if stack[-1] == ')':
                    stack.append(')')
                else:
                    stack.pop()
            else:
                stack.append(')')
    # 左右の括弧がいくつ残っているか記録する
    aN[i] = {'l': stack.count('('), 'r': stack.count(')')}

# どの状態がいくつあるか集計
cnt = {'l': Counter(), 'r': Counter(), 'lr': 0}
for i in range(N):
    if aN[i]['l'] == 0 and aN[i]['r'] == 0:
        cnt['lr'] += 1
    elif aN[i]['l'] == 0:
        cnt['r'][aN[i]['r']] += 1
    elif aN[i]['r'] == 0:
        cnt['l'][aN[i]['l']] += 1

# 集計結果から、いくつペアが作れるか数える
ans = 0
for k, v in cnt['l'].items():
    ans += min(v, cnt['r'][k])
ans += cnt['lr'] // 2
print(ans)
