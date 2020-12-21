"""
参考：https://drive.google.com/file/d/1MAO9xAoPcTmxf8CpvXbw76edm44-XUll/view
・絶対頭いい方法あったんだろうなと思いながら公式解説を確認。
・意外と方針は最初からそう遠くなかった。むしろ最初にやろうとしてた感じは結構近かった。
　題意の操作を言い換えると、全員から-1して1人だけ+2だなー、も考えた。そこから活かせなかった。
・操作回数は最初から一意にできるので、操作回数分最初に全員から引いて、
　そのあと必要な人に+2ずつ配っていく、とやるとすっきりできる。これは確かに。
・まあそれでも書いてみたら結構場合分けあるわ。結局まあまあ大変だ、これ自分で導くの。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for k in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for k in range(b)] for i in range(a)]
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
A = LIST()
B = LIST()

AB = [0] * N
for i in range(N):
    AB[i] = A[i] - B[i]

ans = 0
if N == 2:
    if AB[0] < AB[1]:
        AB[0], AB[1] = AB[1], AB[0]
    ans += -AB[1]
    AB[0] += AB[1]
    AB[1] = 0
    if sum(AB) == 0:
        print(ans)
    else:
        print(-1)
    exit()

if sum(AB) < 0 or sum(AB) % (N-2) != 0:
    print(-1)
else:
    ans = sum(AB) // (N-2)
    for i in range(N):
        AB[i] -= ans
    if max(AB) > 0:
        print(-1)
    else:
        ok = 1
        for i in range(N):
            if -AB[i] % 2 != 0:
                ok = 0
                break
        if ok:
            print(ans)
        else:
            print(-1)
