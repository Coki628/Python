"""
参考：https://twitter.com/50j1_/status/1299710319187521537
・自力ならず。。。
・鳩の巣原理、素因数分解、GCD
・1つ目の条件を満たすには、全てのペアが互いに素でないといけないので、どうチェックするか考える。
・各値を見ていって、1つでも素因数の被るやつがあったらNG、とする。(ここまで分かってたのに)
・方法としては、出現済の素数をsetで持っておいて、各値を愚直に√Aかけて素因数分解して、
　その結果と出現済setを比較、でも間に合う。
・全部やったら100万のNとAへのN√Aなので無理に見えるが、N=100万以下の素数は8万弱くらいしかないので、
　全部の素数を使われたとしても、ちゃんとbreakすれば、この確認は100万回全部やらないで8万くらいで終わる。
・2つ目の条件は全部累積GCDやるだけ。
・素因数分解愚直でも通るならこれは通せるべきだったよなぁ。。。
・ちなみに通ったとはいってもpypyAC1.51秒。想定解は高速素因数分解とのこと。
"""

import sys
from math import gcd

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
INF = 10 ** 19
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def factorize(num: int) -> dict:
    """ 素因数分解 """
    from math import sqrt
    from collections import Counter

    d = Counter()
    for i in range(2, int(sqrt(num))+1):
        while num % i == 0:
            num //= i
            d[i] += 1
        if num == 1:
            break
    if num != 1:
        d[num] += 1
    return d

N = INT()
A = LIST()

def check(se, fact):
    for k in fact:
        if k not in se:
            se.add(k)
        else:
            return False
    return True

is_pc = 1
se = set()
for a in A:
    fact = factorize(a)
    if not check(se, fact):
        is_pc = 0
        break

is_sc = 1
g = 0
for a in A:
    g = gcd(g, a)
if g != 1:
    is_sc = 0

if is_pc:
    print('pairwise coprime')
elif is_sc:
    print('setwise coprime')
else:
    print('not coprime')
