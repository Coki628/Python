"""
・なんとか自力AC、WA叩きまくってダメかと思った。。
・(連続しない)部分列、素数、適切に貪欲
・まず31とかで消せるとしても、3で消して損しないので3,5,7は1回ずつで全部消す。
・1,9だけが残るので、これをどう消すのがいいか考える。19,11はいいけど91,99はダメ、
　911でもいけるけど、それなら11でいけばいい、ってやっていくと、
　19→991→11の順で消せば損しないだろうとなる。ここまで来てAC。
・公式解より、実は19を消した時点で19となる場所がない訳だから9999911111みたいになってる。
　これが考察できてたら、991は順序気にせず数えられるからもっと楽だったな。。
"""

import sys
from collections import Counter

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

N = INT()
S = input()

C = Counter(S)
ans = C['3'] + C['5'] + C['7']

# 19
cnt1 = []
used = [0] * N
for i in range(N):
    if S[i] == '9' and cnt1:
        j = cnt1.pop()
        ans += 1
        used[i] = 1
        used[j] = 1
        C['1'] -= 1
    elif S[i] == '1':
        cnt1.append(i)

# 991
cnt9 = cnt99 = 0
for i in range(N):
    if used[i]:
        continue
    if S[i] == '1' and cnt99:
        cnt99 -= 1
        ans += 1
        C['1'] -= 1
    elif S[i] == '9' and cnt9:
        cnt9 -= 1
        cnt99 += 1
    elif S[i] == '9':
        cnt9 += 1

# 11
ans += C['1'] // 2

print(ans)
