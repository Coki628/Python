"""
参考：https://twitter.com/laycrs/status/1272930971717824513
・自力ならず。。
・周期、GCD、全探索
・内ループはアルファベット分のみで小さいので、「合計何個使うつもりか」を1〜Nまで全探索できる。
・「各ビーズ何個ずつにするか」はGCDするんだけど、これが自分からは出なかった。。
・それが分かったら、各色のビーズについて余り部分を切り捨てる。これ v/x*x みたいにできるんだね。
　(前mod取ったやつ引いたりしてた気がする)
"""

import sys
from collections import Counter
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N, K = MAP()
    S = input()

    C = Counter(S)
    ans = 0
    for ln in range(1, N+1):
        cnt = 0
        x = ln // gcd(ln, K)
        for v in C.values():
            cnt += v // x * x
        if cnt >= ln:  
            ans = max(ans, ln)
    print(ans)
