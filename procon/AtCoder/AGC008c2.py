# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/agc008/editorial.pdf
・自力ならず。。かなり近い所まで来てたのに詰め切れなかった。
・3つ組は2つ作るなら2つ組3つと等価なので、1つ作るかどうかだけが違いになる。
　なので、1つ作るかどうかを両方試せばOK。
・できるだけ2つ組作ってから3つ組作るかその逆の2つでやっててWAだったんだけど、
　これだと、3つ組優先しても3つ組が偶数個になって2つ組全部と変わらなくなってしまう場合があるのでダメっぽい。
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

def check1(I, J, L):
    res = 0
    res += I - I%2
    res += J - J%2
    res += L - L%2
    return res

def check2(I, J, L):
    res = 0
    if I > 0 and J > 0 and L > 0:
        res += 3
        I -= 1
        J -= 1
        L -= 1
    res += I - I%2
    res += J - J%2
    res += L - L%2
    return res

I, O, T, J, L, S, Z = MAP()

ans = O
# 3つ組を使うか使わないかの良い方
ans += max(check1(I, J, L), check2(I, J, L))
print(ans)
