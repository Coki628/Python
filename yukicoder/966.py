"""
参考：https://yukicoder.me/problems/no/966/editorial
　　　https://yukicoder.me/submissions/475335
・自力ならず。。地獄場合分けみたいのしか生えなかった。(WA)
・門松列、貪欲詰め
・大きく分けると条件は2つで、まずaとcが等しいなら-1する。
　次に、bを最小にするか最大にすることを考える。
・口で言うとこれだけだけど、bを最小にする時とかa,c両方適切に減らさないといけないし、
　条件詰めづらかった。(結局人様の提出まで確認した)
・こういう貪欲詰めみたいのって、ほんと成長させにくいよなぁ。。。
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

# bを最小にする
def check1(a, b, c):
    costb = max(b - a + 1, 0)
    b -= costb
    if b > 0:
        return costb
    else:
        return INF

# bを最大にする
def check2(a, b, c):
    costa = max(a - b + 2, 0)
    costc = max(c - b + 1, 0)
    a -= costa
    c -= costc
    if a > 0 and c > 0:
        return costa + costc
    else:
        return INF

for i in range(INT()):
    a, b, c = MAP()

    if a > c:
        a, c = c, a
    ans = 0
    if a == c:
        ans += 1
        a -= 1
    ans += min(check1(a, b, c), check2(a, b, c))
    if ans >= INF:
        print(-1)
    else:
        print(ans)
