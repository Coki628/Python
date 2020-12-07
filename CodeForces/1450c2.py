"""
参考：https://codeforces.com/blog/entry/85348
・C1が解けるとこっちもだいたい同じ。
　今度はXかOに統一する位置を決めてやっぱり3通りやる。
・なんとなく、X,O逆にしたやつも合わせて6通り見ないといけないような気になるけど、
　それはやらなくて大丈夫だった。本番で自分で解くなら、6通り見ちゃうかな。。
"""

import sys

def input(): return sys.stdin.readline().strip()
def list2d(a, b, c): return [[c] * b for i in range(a)]
def list3d(a, b, c, d): return [[[d] * c for j in range(b)] for i in range(a)]
def list4d(a, b, c, d, e): return [[[[e] * d for k in range(c)] for j in range(b)] for i in range(a)]
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

def deepcopy(li): return [x[:] for x in li]

for _ in range(INT()):
    N = INT()
    grid = [[]] * N
    K = 0
    for i in range(N):
        grid[i] = list(input())
        K += grid[i].count('X')
        K += grid[i].count('O')

    for mod in range(3):
        ans = deepcopy(grid)
        cnt = 0
        for i in range(N):
            for j in range(N):
                if grid[i][j] == 'X' and (i+j) % 3 == mod:
                    ans[i][j] = 'O'
                    cnt += 1
                if grid[i][j] == 'O' and (i+j) % 3 == (mod+1)%3:
                    ans[i][j] = 'X'
                    cnt += 1
        if cnt <= K//3:
            for i in range(N):
                print(''.join(ans[i]))
            break
