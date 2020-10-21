"""
参考：https://qiita.com/naru7sh/items/7f3f47fbf2e415c0ec86
・自力AC！
・パスカルの三角形DP
・いつも通りMOD数え上げnCrしようとしたら、素数じゃないMODで割り算できない。。
・先に逆元のやつ覚えたから1度も使ったことなかったパスカルの三角形のやつ、初めてやった。
・1万の2乗=計算量1億のDPはpypyでも色々厳しめ感ありなので、使い回しの配列節約、
　約半分の枝刈りと、万全を期して投げた。無事AC1.94秒。(制約5秒)
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

N = INT()
M = INT()
MOD = 10**9

N //= 1000
rest = N % M

fr = [0] * (rest+1)
to = [0] * (rest+1)
fr[0] = 1
for i in range(M):
    for j in range(rest+1):
        if fr[j] == 0:
            break
        to[j] += fr[j]
        to[j] %= MOD
        if j+1 <= rest:
            to[j+1] += fr[j]
            to[j+1] %= MOD
    fr = to[:]
    to = [0] * (rest+1)
ans = fr[rest]
print(ans)
