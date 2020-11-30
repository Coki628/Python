"""
参考：https://codeforces.com/blog/entry/84885
・自力ならず。upsolveした。
・インタラクティブ、XORの性質、ANDの性質、連立方程式
・まず前提として、a + b = (a ^ b) + 2*(a & b) を知っている必要がある。
　これは、xorするとビットが重なっていない部分だけが残り、andすると重なっている部分だけが残ることから、
　重なっているandした方だけ*2すれば、普通にa+bの和の形になるよ、というもの。
・これで2変数の和が分かるので、3つの変数についてa+b=x,b+c=y,c+a=zみたいな3つの式を立てると、
　連立方程式を解いてa,b,cを導ける。
・まあなんか直感的に、集合をab+ac-bcすれば、2aだけ残るから/2すればaだねって考えてもよさそう。
・後はxorのよくある性質で b = a ^ (a ^ b) みたいに既に分かってるaを打ち消していけば
　任意のbが分かるので全部答え出るよって感じ。
・和を3つ出す時に普通にやるとクエリ6回使っちゃってオーバーするので、
　2番目と3番目のxorは xor23 = xor12 ^ xor 13 みたいに、ここでもxorの打ち消しをうまく使うこと。
　これで最初の3変数を割り出すのに5クエリで済むので、全体がN+2になって間に合う。
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

N = INT()

xor1 = [0] * (N+1)
for i in range(2, N+1):
    print('XOR {0} {1}'.format(1, i), flush=1)
    xor1[i] = INT()
xor23 = xor1[2] ^ xor1[3]
print('AND {0} {1}'.format(1, 2), flush=1)
and12 = INT()
print('AND {0} {1}'.format(1, 3), flush=1)
and13 = INT()
print('AND {0} {1}'.format(2, 3), flush=1)
and23 = INT()
sm12 = 2*and12 + xor1[2]
sm13 = 2*and13 + xor1[3]
sm23 = 2*and23 + xor23

ans = [0] * (N+1)
ans[1] = (sm12 + sm13 - sm23) // 2
ans[2] = (sm12 + sm23 - sm13) // 2
ans[3] = (sm13 + sm23 - sm12) // 2
for i in range(4, N+1):
    ans[i] = xor1[i] ^ ans[1]
print('! {0}'.format(' '.join(map(str, ans[1:]))), flush=1)
