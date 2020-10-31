"""
・なんとか自力AC。
・題意を満たすには、(a+b)-(c+d)がKにならないといけない。この時、a+bを固定すればc+dも一意に定まるので、
　このa+bとc+dが各状態でいくつあるかを数えていく。
・任意の自然数でいいなら、a+bになりうる通り数はa+b-1になる。ここに、a,bはN以下という制約が入る。
　やってみると分かるが、a+bがN+1以下までは上記のルールで増え続け、N+2以降1つずつ減っていくようになる。
　c+dもルールは同じ。あとはこの規則性をうまくコードに落とし込めばいい。
・似たような考察(N以下でa+bにする通り数)を最近ゆきこ埋め(864)でやってたのに、解くのに時間かかりすぎ。。。
・上記の問題から、「N以下の自然数でa+b=xを作る通り数」を返す関数を取ってきたら、
　ほぼそのまま使えて通った。。最初からこれやってれば早かったじゃん。。。
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

# N以下の自然数でa+b=xを作る通り数
def check(x):
    return max(x-1 - max(min(2*(x-1-N), x-1), 0), 0)

N, K = MAP()

ans = 0
for ab in range(-N*2, N*2+1):
    cd = ab - K
    ans += check(ab) * check(cd)
print(ans)
