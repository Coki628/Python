"""
参考：https://twitter.com/50j1_/status/1298299160144142336
・自力ならず。。
・何かを決め打つと線形で決まる貪欲
・そうだとは思ってた。「何か」が分からなくて解けなかった。
・足りなかった視点は、重いやつから取ることに得はないということ。
　言われればそれはそうなんだけど、解いてる最中は設定が頭ん中でグチャグチャになってまとまらなかった。
・軽いのから取る場合だけ考えればいい、が決まれば、
　個数Nの全探索は「自分がi個、相棒がN-i個軽いのを取る」にまとまる。
　重い方は余ったところに持てるだけ持たせればいい。
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
    F, P = MAP()
    N, M = MAP()
    s, w = MAP()

    if s > w:
        s, w = w, s
        N, M, = M, N
    
    ans = 0
    for i in range(N+1):
        cnt = 0
        f, p = F, P
        cs, cw = N, M
        # 自分が軽いやつ(s)を最大i個取る
        use = min(f//s, i)
        f -= use * s
        cs -= use
        cnt += use
        # 相棒が軽いやつを最大N-i個取る
        use = min(p//s, N-i)
        p -= use * s
        cs -= use
        cnt += use
        # 自分が重いやつ(w)をできるだけ取る
        use = min(f//w, cw)
        f -= use * w
        cw -= use
        cnt += use
        # 相棒が重いやつをできるだけ取る
        use = min(p//w, cw)
        p -= use * w
        cw -= use
        cnt += use
        ans = max(ans, cnt)
    print(ans)
