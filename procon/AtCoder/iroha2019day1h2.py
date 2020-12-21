"""
・なんか久し振りに見かけたからやってみた。
・WA叩きまくった末、なんとか自力AC。。
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
sys.setrecursionlimit(10 ** 9)
INF = 10 ** 18
MOD = 10 ** 9 + 7
EPS = 10 ** -10

def digit_sum(n):
    """ 桁和：O(logN) """

    ans = 0
    while n > 0:
        ans += n % 10
        n //= 10
    return ans

N = INT()

# とりあえず桁和
digsm = digit_sum(N)
d, m = divmod(digsm, 9)
# この桁和になる最小値
ans = int(str(m) + str(9) * d)
# 最小値が自分だったら最小から2番目に書き換える
if ans == N:
    tmp = list(str(ans))
    # 1桁または1桁目が9だったら、1とnum-1に分割する
    if len(tmp) == 1 or tmp[0] == '9':
        num = int(tmp.pop(0))
        a, b = str(1), str(num-1)
        tmp = [a] + [b] + tmp
        ans = int(''.join(tmp))
    # それ以外は1桁目を1増やして2桁目を1減らす
    else:
        tmp[0] = str(int(tmp[0])+1)
        tmp[1] = str(int(tmp[1])-1)
        ans = int(''.join(tmp))
print(ans)
