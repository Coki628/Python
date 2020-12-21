"""
参考：https://twitter.com/laycrs/status/1282363059718090753
　　　https://twitter.com/titia_til/status/1282364264603201536
・自力ならず。結構悪くない方針で考察できていたが詰めきれず。。
・消したい区間毎にまとめる。
・バーサークで消せる要素は、いずれも残す両端のどちらかより小さい必要がありそうだが、
　実はどこかで1回でもファイアボールを打つなら、それまでに大きい要素1つに周りを適当に消しておいてもらえば、
　その1回までで大きい要素は全部消すことができる。この考察が出なかった。。
・これが分かると、後はファイアボール0回のパターンだけ両端より全部が小さいかを確認して、
　ファイアボール1回以上なら特に制限なく全部消すを確認、となる。
・こういう、強いやつ1つだけあれば、そいつに他全部消させることで実質強いやつを処理するのは1回だけで済む、
　みたいな考察前にもどっかであったなー。その時も気付けなくて悔しかったんだけど、やっぱ難しいなー。。
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

N, M = MAP()
X, K, Y = MAP()
A = LIST()
B = LIST()

j = 0
tmp = [-INF, []]
C = []
for i in range(N):
    if j != M and A[i] == B[j]:
        tmp.append(A[i])
        C.append(tmp)
        tmp = [A[i], []]
        j += 1
    else:
        tmp[1].append(A[i])
tmp.append(-INF)
C.append(tmp)
if j != M:
    print(-1)
    exit()

ans = 0
for prev, li, nxt in C:
    if li:
        cur = INF
        d, m = divmod(len(li), K)
        for i in range(d+1):
            # ファイアボール0回のパターンだけ両端より全部が小さいかを確認
            if i != 0 or max(li) < max(prev, nxt):
                cur = min(cur, i*X + (d-i)*K*Y + m*Y)
        if cur == INF:
            print(-1)
            exit()
        ans += cur
print(ans)
