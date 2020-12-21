"""
・なんとか自力AC。大変だった。これ本当に300点か。。
・三角形、角度、三角関数、余弦定理
・2辺の長さが分かっていてもう1辺を知りたいので、角度が分かればいいなーって気持ちになる。
・時刻が与えられるが、短針の微妙な位置とか分からんだろ、ってなる。
・ここで先にD解いたり、Eしばらく考えたりして、E厳し目っぽかったので戻ってきた。
・もう1回、300点でそんな高度な要求がある訳ないという前提の元、考え直してみる。
・ここでやっと、1分毎に動く量を出せば経過時間から割り出せると分かる。
・ちょっとゴニョゴニョして無事角度が出て、さて、2辺と角度が揃ったらどうするんだっけな、とググる。
・余弦定理の公式が出てきて、あぁこんなんあったな、と当てはめる。無事AC。
・こういうのを300点っていう低難度で出せちゃうのは、やっぱり競プロは数学ゲーなんだな、と改めて思わされる所。
"""

import sys
from math import cos, radians, sqrt

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

a, b, h, m = MAP()

hm = h*60 + m

# 1周を1として針が今どこにあるか
heach = hm / 720
meach = m / 60
# *360で角度にする
hdig = heach * 360
mdig = meach * 360
# 反対向きのが近い場合を考慮
dig = min(abs(hdig-mdig), 360 - abs(hdig-mdig))

# 2辺と角度から、余弦定理する
ans = sqrt(a**2 + b**2 - (2*a*b*cos(radians(dig))))
print(ans)
