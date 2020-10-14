"""
参考：https://www.youtube.com/watch?v=eus_giFYAIs&feature=youtu.be
・自力ならず。これはまあしゃあない。。
・数学O(1)、数え上げ、対称性
・重複を引いて背反に数え上げる。やっぱこういうのは受験数学やってた人は有利だよね。。
・とりあえず解説動画の方針。でも直線で区切ってa,bを重ならないようにするアプローチは
　なかなか思いつきにくそうだなぁーとも思ったり。思いついても、重複しまくりじゃん、って諦めそう。。
・数え上げる時と重複を省く時で、それぞれ1重と2重にsumのループが回るけど、
　考察する or wolframalpha先生に聞くと、O(1)になる。
　結構雑な形のまま投げたけど、ちゃんと式にしてくれてすげー、ってなった。
・後は正方形なので、対称に4方向のものを考えて*4する。
・愚直でもいいからO(N^2)くらいのΣの式が立てば、そこからO(1)にはできると思うと少しだけ希望が持てた。
・とはいえまあその愚直の式を立てるまでも、頭ついてないと辿り着けないんだけどね。。。
・この手のやつの青diffとか自力で通せるのにはまだまだ先が長そうだね。。
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

# aの位置とbの位置を直線xで区切る
def f1(n, a, b):
    # コーナーケース
    if a+b > n:
        return 0
    # # aの下の辺の位置xを全探索
    # res = 0
    # for x in range(a, n-b+1):
    #     # aの取りうる横位置 * bの取りうる縦横位置
    #     res += (n-a+1) * (n-x-b+1)*(n-b+1)

    # wolframalpha: sum (n-a+1)*(n-x-b+1)*(n-b+1), x=a to n-b
    # -1/2 (a - n - 1) (-b + n + 1) (a + b - n - 2) (a + b - n - 1)
    res = -(a-n-1) * (-b+n+1) * (a+b-n-2) * (a+b-n-1) // 2 % MOD
    return res

# 重複する分を引く
def f2(n, a, b):
    # コーナーケース
    if a+b > n:
        return 0
    # # aの下の辺の位置xと右の辺の位置yを全探索
    # res = 0
    # for x in range(a, n-b+1):
    #     for y in range(a, n-b+1):
    #         # bの取りうる縦横位置
    #         res += (n-x-b+1)*(n-y-b+1)

    # wolframalpha: sum((n-x-b+1)*(n-y-b+1)) for y=a to n-b for x=a to n-b
    # 1/4 (a + b - n - 2)^2 (a + b - n - 1)^2
    res = (a+b-n-2)**2 * (a+b-n-1)**2 // 4 % MOD
    return res

for _ in range(INT()):
    N, a, b = MAP()

    # それぞれ4方向に対称なので*4する
    ans = (4*f1(N, a, b) - 4*f2(N, a, b)) % MOD
    print(ans)
