# -*- coding: utf-8 -*-

"""
参考：https://drken1215.hatenablog.com/entry/2019/06/16/020500
・自力ならず。
・効率よく全探索
・公式はDPしてたけど、けんちょんさんの全探索解が自分がやろうとしてた方向と近かったので、
　とりあえずそっちでやった。
・金銀銅それぞれ前半後半どっちかしかやらない、は合ってて、
　有りうる分け方は2つ決め打つと残りで3つ目も定まるからN^2で全探索、も合ってた。
・見落としてたのは、前半の換金で得たドングリは後半の換金に使えるってこと。
　なんかそれぞれ独立にしかできない気がしてたんだけど、確かに使えるよね。。
・後は金銀銅をそれぞれ前半に換金するか後半に換金するかで予め分けておいて、
　その分け方に従って全探索するようにすればOK。
・N^2しなくちゃいけない3つへの振り分けは前後半のどちらかでしか行われなくて、
　その時の反対側は何もしない、になるので大丈夫。
・計算量はN^2=5000^2=2500万で、pypyAC1.6秒。
"""

import sys
from itertools import accumulate

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
A = LIST()
B = LIST()

# 金銀銅を前後半どっちでやるか振り分け
S1 = []
S2 = []
for i in range(3):
    if A[i] < B[i]:
        S1.append(i)
    else:
        S2.append(i)
        # 後半でやる方は処理を共通化するため向きを逆にしておく
        A[i], B[i] = B[i], A[i]

# cnt個のドングリをi(金銀銅)で換金する時に得られるドングリの数を返す
def calc(cnt, i):
    d, m = divmod(cnt, A[i])
    return d * B[i] + m

# S := 金銀銅のうち、今回換金するものの集合
# N := 今回使えるドングリの数
def solve(S, N):
    # 何種類やるかでループの深さとか変わるので場合分け
    if len(S) == 0:
        return N
    elif len(S) == 1:
        return calc(N, S[0])
    elif len(S) == 2:
        res = 0
        # N個をx,yの2グループに振り分けるを全探索
        for x in range(N+1):
            y = N - x
            if y < 0:
                break
            cnt = calc(x, S[0]) + calc(y, S[1])
            res = max(res, cnt)
        return res
    else:
        res = 0
        # N個をx,y,zの3グループに振り分けるを全探索
        for x in range(N+1):
            for y in range(N+1):
                z = N - x - y
                if z < 0:
                    break
                cnt = calc(x, S[0]) + calc(y, S[1]) + calc(z, S[2])
                res = max(res, cnt)
        return res

# 前半(取引所A→B)
N2 = solve(S1, N)
# 後半(取引所B→A)
ans = solve(S2, N2)
print(ans)
