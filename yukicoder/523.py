"""
・自力AC！
・数え上げ、重複を含む順列
・これ系の数え上げだいたい水diff解けないから通せて嬉しい。
・2N個のLEDがあって、それを並べる順列、だけど1回目と2回目は入れ替えれないから、
　それぞれ2つは区別をなくす、って考えると、(2*N)! / 2^N　が出てくる。
　後はこれをMOD階乗の下準備してコードに落とすだけ。
・MOD階乗の下準備パートで100万くらいは今までもあったけど、
　Nが1000万なのは初めてだったので、間に合うかちょっと心配だったけど大丈夫だった。pypyAC1.04秒。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c for j in range(b)] for i in range(a)]
def list3d(a, b, c, d): return [[[d for k in range(c)] for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e for l in range(d)] for k in range(c)] for j in range(b)] for i in range(a)]
def ceil(x, y=1): return int(-(-x // y))
def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
def Yes(): print('Yes')
def No(): print('No')
def YES(): print('YES')
def NO(): print('NO')
sys.setrecursionlimit(10**9)
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

class ModTools:
    """ 階乗・逆元用のテーブルを構築する """

    def __init__(self, MAX, MOD):

        # nCrならn、nHrならn+rまで作る
        MAX += 1
        self.MAX = MAX
        self.MOD = MOD
        factorial = [1] * MAX
        factorial[0] = factorial[1] = 1
        for i in range(2, MAX):
            factorial[i] = factorial[i-1] * i % MOD
        inverse = [1] * MAX
        inverse[MAX-1] = pow(factorial[MAX-1], MOD-2, MOD)
        for i in range(MAX-2, -1, -1):
            inverse[i] = inverse[i+1] * (i+1) % MOD
        self.fact = factorial
        self.inv = inverse

    def div(self, x, y):
        """ MOD除算 """

        return x * pow(y, self.MOD-2, self.MOD) % self.MOD

N = INT()

mt = ModTools(2*N+1, MOD)
ans = mt.div(mt.fact[2*N], pow(2, N, MOD))
print(ans)
