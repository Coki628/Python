"""
参考：色々な人たちのツイートとか
・すごく理解に時間かかったけど、やっと分かった。
　最適な数列はどう作ればいいか、が分かると方針が見えてくる。
・まず数列は昇順に並んでおり、単調増加である。これをうまく使うと、数列Aを進めていく際に、
　「数列Bはほぼ全部変更なしで、1箇所だけ、ひとつ前との差分だけ増やす。」
　という方針を取り続けることができる。こうすると、数列Aの値が変化した時、数列Bで種類数を
　増やす必要がある箇所は、常に1箇所に抑えることができる。
・あとは数列Bを増やす位置を、毎回均等に持ち回るようにすれば、全体として数列Bの種類数最小を達成できる。
・これを数列Bが1〜Nまでを全部試して、無理なら-1、K以下に抑えたらそこが答え。
・愚直に実装したら、これはさすがにループが1つ深いのでTLE。。
　でもこの考え方を使いつつ、数列Bを陽に持たずに、今種類数が何個になったかだけ持つようにすれば
　ループが1つ減って間に合う。
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
INF = 10**19
MOD = 10**9 + 7
EPS = 10**-10

for _ in range(INT()):
    N, K = MAP()
    A = LIST()

    ans = INF
    for i in range(1, N+1):
        B = [[] for j in range(i)]
        # 最初は適当に先頭に置く、でOK
        B[0].append(A[0])
        for k in range(1, i):
            B[k].append(0)
        l = 1
        for j in range(1, N):
            for k in range(i):
                if k == l%i:
                    # 今回はこの位置を増やす
                    B[k].append(B[k][-1]+(A[j]-A[j-1]))
                else:
                    # 他の位置はそのまま
                    B[k].append(B[k][-1])
            # 差分があった場合は次に増やす位置を進める
            if A[j]-A[j-1] != 0:
                l += 1 
        mx = 0
        for k in range(i):
            mx = max(mx, len(set(B[k])))
        if mx <= K:
            ans = min(ans, i)
            break
    if ans == INF:
        print(-1)
    else:
        print(ans)
