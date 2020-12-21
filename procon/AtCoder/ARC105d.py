"""
・自力ならず。。
・ゲーム系、Nim応用
・題意のゲームの後半部分はNimになる。よってそこまでに全部のxorが0になるかどうかが重要になる。
・全部の袋を皿に移した時点からNimが始まるので、このNim開始時に先手番になった方は0にしたくない、
　後手番は0にしたい、という状況になるため、両者はゲームの前半これを目指して動く。
・ここまで考えて、ぱっと思ったところは、0にする方が大変そうで、0じゃなくする方が楽そうだな、と。
　実際、この印象は合っていて、0になる条件はかなり限定的で、その場合以外は0じゃなくする方が勝ちになる。
・この方針までは来ていたけど、0になる条件が正しいところまで詰められなかった。。
・正解としては、同じ数字が全て偶数個あればOK、そうでないなら全部ダメ、になる。
　0にしたい方の唯一の勝利方針として、直前に相手が取ったやつと同じものを取り続ける、というものがあるため。
　逆にこの戦略が取れない場合は、0にしたくない方がどうにでも状況を崩せるので勝つ。
　具体的には、なるべく大きい数からくっつけてしまえば、
　相手はどうやってもその大きい数が持つビットを0にできないという状況になるため。
・もうちょっとだった気になってるけど、こういうゲーム考察はなんかうまくいくか運ゲーなところが
　大きいと思うから、こっちが解けなくてもCが通せるくらいの実力をつけたいね。。
"""

import sys
from collections import Counter

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

for _ in range(INT()):
    N = INT()
    A = LIST()

    swapped = (N % 2 == 1)
    C = Counter(A)
    is_0 = True
    for v in C.values():
        if v % 2 != 0:
            is_0 = False
            break

    if swapped ^ (not is_0):
        print('First')
    else:
        print('Second')
