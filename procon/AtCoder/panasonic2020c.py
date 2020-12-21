# -*- coding: utf-8 -*-

"""
参考：https://img.atcoder.jp/panasonic2020/editorial.pdf
　　　https://twitter.com/TT_beginner/status/1238822286246854657
・自力ならず。。誤差だろうは思って、EPSとか頑張って入れてみたけどダメだった。
・誤差、精度、式変形
・公式解
・式変形で整数にする。
・それはちょっと考えたんだけど、(√a+√b)^2 < c までやって、
　(a+b)^2=a^2+2ab+b^2 っていう基本過ぎる展開を出せなくて終了。。
　言われたら思い出したけど、何で出てこなかったんだろう。。
・でも仮にこれができていてもまだ罠があって、多分通せてない。
　最終的に 4ab < (c-a-b)^2 って形になるんだけど、この式変形はc-a-bが負になってしまうと、
　√内がマイナスみたいなことになって破綻するので、そこは別処理しないといけない。
・数学とはいえ言われれば分かる基本的な知識なのに、出てこないのつらいなー。。
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

a, b, c = MAP()

if c-a-b >= 0 and 4*a*b < pow(c-a-b, 2):
    Yes()
else:
    No()
