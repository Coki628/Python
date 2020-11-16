"""
・なんとか自力AC
・最初の方、直前位置の辺の状態を持ってDP、みたいなことやろうとして死んだ。。
　その後は位置の偶奇で状態が決まったりする気がしてこれまた迷走。。
　全くいい方針に辿り着かず、実験コード書いて何とか見えてきた方針から、何とか通したって感じ。。。
・結局、偶数個の切り替えは自由にできるので、マイナスになってしまう場所は最大1個しかない。
　そうなってしまう条件は、ゼロの場所がなく、マイナス(プラスも同じ)の場所が奇数個になること。
　ゼロが1つでもあればそこを使えば帳尻合わせできる。ないとしたら、全体は長方形なので絶対偶数個で、
　マイナスが奇数個ならプラスも奇数個。この中でどれか1つは符号がマイナスになってしまう。
　このマイナスにする場所は、絶対値がなるべく小さい場所がいいので、
　全体を数えながら絶対値の最小値も取っておく。
・これ普通にむずくないかね。。。どっか1箇所以外は適当にやればうまくいくって事実は
　みんなどこで気付くんだろ。実験するまで全然見えなかったよ。。
・さらにシンプルに、絶対値の最小値取った時に0があれば
　減らさないといけない数が勝手に0になるから、0があるかは持ってなくてもよかった。。
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

for _ in range(INT()):
    H, W = MAP()
    grid = [[]] * H
    for i in range(H):
        grid[i] = LIST()

    mcnt = 0
    mn = INF
    ans = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] < 0:
                mcnt += 1
            mn = min(mn, abs(grid[i][j]))
            ans += abs(grid[i][j])
    if mcnt % 2 == 1:
        ans -= mn*2
    print(ans)
