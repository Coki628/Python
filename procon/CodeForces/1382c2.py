"""
・自力AC！
・基本方針はC1と同じ。ボトルネックだったNかかる全体XORと全体リバースをなんとかする。
・何回フリップしたかと何回リバースしたかの偶奇を持っておく。
　これで陽に操作しなくても、次にどこを動かしたらいいか分かるようになる。
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
INF = 10 ** 19
MOD = 10 ** 9 + 7

for _ in range(INT()):
    N = INT()
    S = list(map(int, input()))
    T = list(map(int, input()))

    ans = []
    l = 0
    r = N - 1
    flip = 0
    rev = 0
    for i in range(N-1, -1, -1):
        if not rev:
            if S[l]^flip == T[i]:
                if l >= r:
                    break
                ans.append(1)
                S[l] ^= 1
            l += 1
        else:
            if S[r]^flip == T[i]:
                if l >= r:
                    break
                ans.append(1)
                S[r] ^= 1
            r -= 1
        ans.append(i+1)
        flip ^= 1
        rev ^= 1
    ans.insert(0, len(ans))
    print(*ans)
