"""
参考：https://yukicoder.me/problems/no/933/editorial
　　　https://scrapbox.io/ganariya/YukicoderContest232_B%e5%95%8f%e9%a1%8cP1.5_%e3%80%8c%e3%81%8a%e3%81%be%e3%82%8f%e3%82%8a%e3%81%95%e3%82%93%e3%81%93%e3%81%84%e3%81%a4%e3%81%a7%e3%81%99%e3%80%8d
・自力ならず。
・数字根
・これは知らなかったからしょうがない。mod 9すると求められる。0の時は場合分け。
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
EPS = 10 ** -10

N = INT()
A = LIST()

if A.count(0):
    print(0)
    exit()

mod = 9
ans = 1
for a in A:
    ans *= a
    ans %= mod
if ans == 0:
    print(mod)
else:
    print(ans)
