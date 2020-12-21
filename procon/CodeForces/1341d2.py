# -*- coding: utf-8 -*-

"""
参考：皆様のツイート
・DP、復元、ビットの管理、貪欲
・dp[i][j][k] := i番目まで見て、棒をj個使った状態で、最後に使った数字がkの状態に到達可能か
　みたいなDPの3次元目をビット管理にする。
・これを予め桁の順番逆にして、末尾桁から先頭桁に向かってのDPにしておくと、
　終了から遡る時に先頭桁から貪欲に(大きい数字があれば取るように)選んでいける。
・復元パートについては、その時選ぶ数字が決まれば、その数字に必要な棒の数 - その桁に元々あった棒の数
　で、その時何本の棒を使ったかが分かるので、DP上どの位置に戻るのかが一意に定まる。
　到達できた所から遡っているので、途中で行き止まりってこともない。
・なお、計算量改善のため、使う棒の7ビット分見比べる所は1度で全部見れるようにした。
　具体的には、
　・Aが立っててBが立ってないところがあればその数字には遷移不可
　・Bが立っててAが立ってないところの数だけ棒を使う
　の2点が分かればいいので、以下はベン図的なものを書いたら分かったんだけど、
　A全体 - AかつBのところ で前者、B全体 - AかつBのところ で後者になる。
・あとそれでもTLEしてたんだけど、popcountを文字列使わないやつにしたら通った。
　やっぱ数千万単位の回数やるとなると、数値のままやる方が全然速いみたい。
・計算量は2000*2000*10=4000万、pypyAC0.5秒。
"""

import sys
from itertools import accumulate

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
INF = 10 ** 18
MOD = 10 ** 9 + 7

def popcount(i):
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    i = (i + (i >> 4)) & 0x0f0f0f0f
    i = i + (i >> 8)
    i = i + (i >> 16)
    return i & 0x3f

B = [
    '1110111', '0010010', '1011101', '1011011', '0111010', '1101011', '1101111', '1010010', '1111111', '1111011',
]
for i in range(10):
    B[i] = int(B[i], 2)

N, K = MAP()
A = [int(input(), 2) for i in range(N)][::-1]

dp = list2d(N+1, K+1, 0)
dp[0][0] = 1
for i, a in enumerate(A):
    for j in range(K+1):
        if not dp[i][j]:
            continue
        for x, b in enumerate(B):
            ab = a & b
            cnt = popcount(b ^ ab)
            ng = 1 if (a ^ ab) > 0 else 0
            # for k in range(7):
            #     if not a>>k & 1 and b>>k & 1:
            #         cnt += 1
            #     if a>>k & 1 and not b>>k & 1:
            #         ng = 1
            #         break
            if not ng and j+cnt <= K:
                dp[i+1][j+cnt] |= 1 << x

if not dp[N][K]:
    print(-1)
    exit()

cnt = K
ans = []
for i in range(N, 0, -1):
    for j in range(9, -1, -1):
        if dp[i][cnt] >> j & 1:
            num = j
            break
    ans.append(str(num))
    need = popcount(B[num]) - popcount(A[i-1])
    cnt -= need
print(''.join(ans))
