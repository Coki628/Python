# -*- coding: utf-8 -*-

"""
参考：https://www.slideshare.net/chokudai/indeednow-quala
・公式解説より
・考察力。頭いい。にぶたん要らなかった。。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for j in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7

N = INT()
A = LIST(N)
A.sort(reverse=1)

# 0点の参加者を省く
for i in range(N):
    if A[i] == 0:
        A = A[:i]
        break
N = len(A)

for i in range(INT()):
    cap = INT()
    # 全員入るのでボーダーは0でOK
    if cap >= N:
        print(0)
    else:
        # 0から数えてcap人目がギリギリ通らないように設定する
        ans = A[cap] + 1
        print(ans)
