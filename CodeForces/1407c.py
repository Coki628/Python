"""
参考：https://twitter.com/kumakumaaaaa__/status/1303373023160926209
・自力ならず。。。
・インタラクティブ、MODの特性
・できなかった。1つ固定して他と比較していって、全部見ればとりあえずその1つは分かるけど、、
　とか、最大のやつが求まればそいつを軸にして他が全部分かるはず、とか、色々考えたんだけど。。
・正解は、同じペアで向き逆でやる、だった。これやると、小さい方は必ず答えが出る。
　言われたらああ確かに、だな。。
・実装上は、前後から始めて、確定したところから狭めていく感じにした。
　最大値だけ最後までブランクなので別で埋める。
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
INF = 10**19
MOD = 10**9 + 7

N = INT()

ans = [0] * (N+1)
i = 1
j = N
while i < j:
    print('? {0} {1}'.format(i, j), flush=1)
    k1 = INT()
    print('? {0} {1}'.format(j, i), flush=1)
    k2 = INT()
    if k1 > k2:
        ans[i] = k1
        i += 1
    else:
        ans[j] = k2
        j -= 1
for i in range(1, N+1):
    if ans[i] == 0:
        ans[i] = N

print('! {0}'.format(' '.join(map(str, ans[1:]))), flush=1)
