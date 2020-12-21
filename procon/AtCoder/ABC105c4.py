# -*- coding: utf-8 -*-

"""
・速解き特訓ABC周回
・前の自分のやつ参考にちょっと変更。やっぱりこっちのが公式解より分かるかな。
・N進数はNで割れば桁を落とせる、は他の場合でも使えそう。
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

a = INT()

if a == 0:
    print(0)
    exit()

ans = ''
while abs(a) > 0:
    # 小さいビットから見て行って、その時点の偶奇で値を特定していく
    if a % 2 == 1:
        # 今の末尾の桁は1だったので、その分を引く
        a -= 1
        ans += '1'
    else:
        ans += '0'
    # -2進数は-2で割れば桁を落とせる(10進数で割る10する感じ)
    a /= -2
print(ans[::-1])
