# -*- coding: utf-8 -*-

"""
参考：https://www.ioi-jp.org/camp/2014/2014-sp-tasks/2014-sp-d3-joioji-review.pdf
　　　https://tinumu.github.io/diary/15.html
・自力ならず。。考察強い。結局実装の方針も人のやつ見たし。。
・累積和、式変形
・普通の累積和を式変形：J[t] - J[s-1] = O[t] - O[s-1] -> J[t] - O[t] = J[s-1] - O[s-1]
　これで開始地点と終了地点を独立に扱える。j - o = jo, j - i = ji とする。
・(jo, ji)のペアで現在の状態を持っておいて、出現済のペアが来たら、最初に出た時のindexと比較していくことで、
　r - l が最大になる場所をO(N)で見つけられる。
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

# jo := j - o, ji := j - i として動的に更新していく
jo = 0
ji = 0
# D[(a, b)] := joの値がa、jiの値がbの状態となった最初のindex
D = {(0, 0): -1}
ans = 0
for i, s in enumerate(S):
    if s =='J':
        jo += 1
        ji += 1
    elif s == 'O':
        jo -= 1
    else:
        ji -= 1
    # 既に出現済のペアが来たら、最初に出現したindexとの距離を取る
    if (jo, ji) in D:
        ans = max(ans, i - D[(jo, ji)])
    else:
        D[(jo, ji)] = i
print(ans)
