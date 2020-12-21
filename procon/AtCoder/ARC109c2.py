"""
・なんとか自力AC。。
・大変だった。色々実験して、とりあえずRPSRPSって規則的に続くやつ以外は
　じきに何か1つに収束してそう、ってなったので、そこだけ場合分けして
　あとは適当に12回くらいやって答えにする、ってやっててWA。。
・色々もがいて、最終的に適当にやる回数を22回まで増やしたらAC。。
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

def check(a, b):
    if a == b:
        return a
    if a == 'R' and b == 'S' \
            or a == 'S' and b == 'P' \
            or a == 'P' and b == 'R':
        return a
    else:
        return b

N, K = MAP()
S = input()

MX = 22
sz = 1<<MX
S2 = []
for i in range(sz):
    S2.append(S[i%N])

if N == 1:
    print(S[0])
    exit()
if N == 2:
    ans = check(S[0], S[1])
    print(ans)
    exit()

if K <= 12:
    res = [[] for i in range(K+1)]
    for i in range(1<<K):
        res[0].append(S2[i])
    for i in range(1, K+1):
        for j in range(0, len(res[i-1]), 2):
            a = res[i-1][j]
            b = res[i-1][j+1]
            winner = check(a, b)
            res[i].append(winner)
    ans = res[K][0]
    print(ans)
    exit()

res = [[] for i in range(MX+1)]
for i in range(sz):
    res[0].append(S2[i])
for i in range(1, 2):
    for j in range(0, len(res[i-1]), 2):
        a = res[i-1][j]
        b = res[i-1][j+1]
        winner = check(a, b)
        res[i].append(winner)

flag = 0
base = res[1][:3]
if len(set(base)) == 3:
    for i in range(sz//2):
        if res[1][i] != base[i%3]:
            flag = 1
            break
else:
    flag = 1

for i in range(2, MX+1):
    for j in range(0, len(res[i-1]), 2):
        a = res[i-1][j]
        b = res[i-1][j+1]
        winner = check(a, b)
        res[i].append(winner)

if flag:
    ans = res[MX][0]
    print(ans)
else:
    table = [[]] * 6
    for i in range(6):
        table[i] = res[i+1][:3]
    ans = table[(K-1)%6][0]
    print(ans)
