"""
・ここまで自力AC！
・数学系、数え上げ、nCr系、重複度で割る、円順列
・みんなもっと爆速で解いてたけど、自分的にはこういう数学O(1)な数え上げを自力で通せたのは嬉しい。
　マスターオブ場合の数で学んだノウハウを総動員した感じだ。
・まずN人をN/2人ずつのグループに分けたいけど、これはN個からN/2個選ぶのnCrをすればよくて、
　でもこれだと2つのグループを区別してしまっているので、同じ組になってる分で、/2する。
　(これグループ2つだったからよかったけど、3つ以上だと話変わりそう。。)
・ここまでは割と早かったんだけど、答え合わず。なんでだー、、ってなってて、
　条件よくよく見たら、同じグループになってる人たちは完全に並び順無視なんじゃなくて、
　回転させて同じになるやつは同じとみなす、だった。
・これはいわゆる円順列ってやつで、1人固定して(N-1)!ってやるといいって話がある。
　今回は1つのグループにN/2人いるので、(N/2-1)!で、この並び順の区別が両方のグループで発生するので、
　ここは2つになって((N/2-1))!^2になる。これをさっきのグループ分けの分と掛け合わせればOK。
"""

import sys
from math import factorial

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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def nCr(n, r):
    from math import factorial

    if n < r: return 0
    # 10C7 = 10C3
    r = min(r, n-r)
    return factorial(n) // (factorial(r) * factorial(n-r))

N = INT()
ans = nCr(N, N//2) * factorial(N//2-1)**2 // 2
print(ans)
