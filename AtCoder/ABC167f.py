"""
参考：https://www.youtube.com/watch?v=ENSOy8u9K9I&feature=youtu.be
・括弧系
・頭いい。上りと下りで分けて考える。
・上りはなるべく右が少ないのから、下りは反対なので、左がなるべく多いのから。
・解法聞いてなんかデジャヴ感があった。ツイッターにも上がってた。ARC053Cだ。
　題材は違うけど、行きと帰りでの貪欲の決め方とか、考え方は同じだね。
　解いたことあるやつだけど、なかなかやってる最中に結びつかないよなぁ。。
"""

import sys
from operator import itemgetter

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
A = []
B = []
cur = 0
for i in range(N):
    S = input()
    tmp = []
    for s in S:
        if s == '(':
            tmp.append(s)
        else:
            if tmp and tmp[-1] == '(':
                tmp.pop()
            else:
                tmp.append(s)
    l = tmp.count('(')
    r = len(tmp) - l
    # 上りに使うか下りに使うかで分けておく
    if r-l <= 0:
        A.append((l, r))
    else:
        B.append((l, r))
# 行きは)が少ないのを優先
A.sort(key=itemgetter(1))
# 帰りは(が多いのを優先
B.sort(key=itemgetter(0), reverse=1)

for l, r in A:
    cur -= r
    if cur < 0:
        No()
        exit()
    cur += l
for l, r in B:
    cur -= r
    if cur < 0:
        No()
        exit()
    cur += l
if cur == 0:
    Yes()
else:
    No()
