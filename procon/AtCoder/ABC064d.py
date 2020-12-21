# -*- coding: utf-8 -*-

"""
・結構さくっと自力AC
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)

from collections import deque

N = int(input())
S = input()

ans = deque()
tmp = ''
l = r = 0
# 基本方針は、追加する時は(は左端、)は右端に足すこと
for c in S:
    # (が来たらとりあえずキープ
    if c == '(':
        l += 1
        tmp += '('
    else:
        if l == 0:
            # (がない時に)が来たらペアに出来ない確定なので(追加
            ans.appendleft('(')
            ans.append(')')
        else:
            r += 1
            tmp += ')'
            if l == r:
                # )が(と同じだけ集まったらペア確定
                ans.append(tmp)
                tmp = ''
                l = r = 0
if l > r:
    # 最後に余った(に対応する)を追加
    tmp += ')' * (l-r)
    ans.append(tmp)
print(''.join(ans))
