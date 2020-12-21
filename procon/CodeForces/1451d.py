"""
参考：https://codeforces.com/blog/entry/84885
・ゲーム系
・そもそも誤読して問題難しくしてた。。x側+y側=kになるように自由に進めるのかと思ったら、
　x側にkかy側にkの2択だね。。道理でむずいと思った。。。
・まあそれでも正解の考察ができたかと言われると。。
　でも2択だったら、愚直シミュ書いて規則を調べるくらいのことはできたはず。。
・結果的にd以内になる一番遠い座標(x*k,x,k)を考えた時、ここから1歩動ければ
　先手番の勝ち、動けなければ後手番の勝ちになる。
・どうしてそうなるかを考えると、もしここから動けない場合、後手番は自分のターンで常にこの
　(x*k,x*k)にコマを持ってこれるので勝てる。
　動ける場合、最初のターンで先手番が1つ動いた状態が、ちょうど上の後手番の状態の反対みたいになるので、
　今度は先手番が勝ち位置を保てる。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

def bisearch_max(mn, mx, func):
    ok = mn
    ng = mx
    while ok+1 < ng:
        mid = (ok+ng) // 2
        if func(mid):
            ok = mid
        else:
            ng = mid
    return ok

def check(m):
    return (m*k)**2 + (m*k)**2 <= d**2

for _ in range(INT()):
    d, k = MAP()

    m = bisearch_max(0, d, check)
    x = m*k
    y = x
    if x**2 + (y+k)**2 <= d**2:
        print('Ashish')
    else:
        print('Utkarsh')
