"""
・自力AC！
・ゲーム系
・半信半疑でのぶん投げだけど通って嬉しい。
・とりあえずゲームの進行を考えると、一番でかい山がかなりでかい場合、先行はそこを使って
　どうにでも回数調整できてしまうので負け無しなのでは、となる。
・具体的には、他の山全部合わせたより大きければ、最後まで帳尻合わせができるだろう、と。
・で、それが無理な場合、2人とも1個ずつしか取れないルール的に、単に合計の偶奇で決まりそう、となる。
・全然証明とか無理なんだけど、通ったから正義。。。
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
    A = LIST()

    mx = max(A)
    sm = sum(A)
    if mx > sm - mx:
        print('T')
    else:
        if sm % 2 == 1:
            print('T')
        else:
            print('HL')
