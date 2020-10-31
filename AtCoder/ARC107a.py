"""
・なんとか自力AC。相変わらず数学弱すぎて泣けてくる。
・wolframalpha、エスパー
・重なり合うシグマを見て脳が死んで、wolframalphaに投げたら正しく読み込まず、
　ループを1つ減らして「sum(a*b) for a=1 to A for b=1 to B」ってやったら出て、
　ここからエスパーして通した。我ながらひどい。
参考：https://twitter.com/prd_xxx/status/1322539572144664577
・式変形。掛け算はシグマの外側に出せるので、ΣΣΣabcはΣaΣbΣcみたいな形にしていい。
　これならただの公差1の等差数列の和*3つで、(n+1)C2を3つ掛けるだけになる。。
参考：https://mathtrain.jp/wolalpha
・さらに追記。書き方変えたらwolframalphaにちゃんと出る。
　「sum[sum[sum[a*b*c, {c, 1, C}], {b, 1, B}], {a, 1, A}]」って書いたらちゃんと3重Σとして認識した。。
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
MOD = 998244353
EPS = 10**-10

A, B, C = MAP()

ans = A*(A+1)*B*(B+1)*C*(C+1) // 8 % MOD
print(ans)
