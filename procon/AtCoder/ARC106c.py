"""
・なんとか自力AC！
・区間スケジューリング応用、構築
・区間スケジューリングの考え方に基づくと、右端ソート貪欲が最善で左端ソート貪欲はだめになる。
　左端ソートがだめになるような場所に適切に必要な個数の区間を配置すればOK。
・基本的に差分MがN-2以下じゃないとダメだけど、N=1の場合はM=0でもうまくいってしまうのでコーナーケース。
　1WAで絶望しながら極端ケース適当に試してて、たまたま発見できたからよかった。。
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

N, M = MAP()

# コーナーケース
if N == 1 and M == 0:
    print(1, 2)
    exit()

if M < 0 or M >= N-1:
    print(-1)
    exit()

L = 10**6
ans = []
# 左端ソートが取る最初の区間
ans.append((1, L))
# 右端ソートが取る最初の区間
ans.append((2, 3))

# 右端ソートしか取れない場所にM個の区間を置く
x = 4
for _ in range(M):
    ans.append((x, x+1))
    x += 2

# 2人とも取れる場所に残りを置く
x = L + 1
for _ in range(N-M-2):
    ans.append((x, x+1))
    x += 2

for a in ans:
    print(*a)
