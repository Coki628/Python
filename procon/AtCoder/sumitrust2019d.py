# -*- coding: utf-8 -*-

"""
参考：https://twitter.com/maspy_stars/status/1201134356632203267
・自分で出なかった全探索解。1000通り全部調べる。
・確かに計算量1000*3万=3000万だから0から999まで全部をNかけて調べても良かったんだよな。
　言われたらはっとしたけど、先にDPを考えてた頭でこっちの発想に持っていくのが難しかった。。
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
S = input()

def check(x):
    # 探す桁i
    i = 0
    for a in S:
        # 探してる数字が見つかったら使う
        if a == x[i]:
            i += 1
            # 3桁揃ったらOK
            if i == 3:
                return True
    return False

ans = 0
# 0〜999まで全部見る
for i in range(10):
    for j in range(10):
        for k in range(10):
            x = str(i) + str(j) + str(k)
            if check(x):
                ans += 1
print(ans)
