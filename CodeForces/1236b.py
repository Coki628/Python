# -*- coding: utf-8 -*-

"""
・自力AC！
・組み合わせ系
・今回は箱に制限はないが、プレゼントは各1回必ず使う制約で、1度も使わないパターンをどう除くか悩んだ。
・箱にプレゼントを割り当てる方針でなく、プレゼントをどの箱に使うかの方針で考えると見通しが良くなった。
・nCn..0の全組み合わせは2^N。まあ、N個について使うor使わないだからそれはそうなんだけど、
　先にnCrが頭にあったから辿り着くのが大変だった。
・時間かかったけど、こういう数学系自分で通せたから嬉しい。
・とはいえこれ、コンテスト中3000人以上通してるくらいだから、本当はもっと速解きしたい。
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
INF = float('inf')
MOD = 10 ** 9 + 7

N, M = MAP()

print(pow(pow(2, M, MOD)-1, N, MOD))
