# -*- coding: utf-8 -*-

"""
・速攻自力AC
・想定外のケースがあるかと思えば素直に通った。
・これ500点でいいのだろうか。
"""

import sys
def input(): return sys.stdin.readline().strip()
sys.setrecursionlimit(10 ** 9)
INF = float('inf')
MOD = 10 ** 9 + 7

N, A, B = map(int, input().split())
xN = list(map(int, input().split()))

ans = 0
for i in range(1, N):
    ans += min((xN[i] - xN[i-1]) * A, B)
print(ans)
