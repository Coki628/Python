"""
参考：誰かのツイートとか
・あまりにも地獄実装だったので、ちょっと人々の話を聞いて改善。
・大きい変更としては、2*2マスのチェックは、L字の置き方4つの置く/置かない=2^4=16通りをビット全探索。
　これでとりあえず地獄の場合分けはやらなくて済む。
　H*W<=2万っていう中途半端な制約はこのビット全探索を想定してたのかな。。
・あとグリッド転置して縦横を省略ね。これも典型なのに愚直してた。。
・他にもちょこちょこいじったのはソース内コメント参照。
・あとそもそも、左上から右下に寄せていって、右下2*2に集めてそこだけちゃんとやる、
　みたいな方法もあったらしくて楽だったぽい。みんな賢い。。。
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

def check1(i, j):
    # 置き方パターン4通り
    pattern = [[(i, j), (i+1, j), (i, j+1)],
               [(i+1, j), (i, j+1), (i+1, j+1)],
               [(i, j+1), (i+1, j+1), (i, j)],
               [(i+1, j+1), (i, j), (i+1, j)],
    ]
    # 置くか置かないかビット全探索
    for bit in range(1<<4):
        res = []
        for k in range(4):
            if bit>>k & 1:
                row = []
                for h, w in pattern[k]:
                    grid[h][w] ^= 1
                    row += [h, w]
                res.append(row)
        cnt = 0
        for h in range(i, i+2):
            for w in range(j, j+2):
                cnt += grid[h][w]
        # 状態を戻す
        for k in range(4):
            if bit>>k & 1:
                for h, w in pattern[k]:
                    grid[h][w] ^= 1
        # 全部消えてたらこれでOK
        if not cnt:
            return res
    raise Exception

def check2(i, j):
    res = []
    if grid[i][j] and grid[i-1][j]:
        res.append([i, j, i, j-1, i-1, j-1])
        res.append([i-1, j, i, j-1, i-1, j-1])
    elif grid[i][j]:
        res.append([i, j, i, j-1, i-1, j-1])
        grid[i][j-1] ^= 1
        grid[i-1][j-1] ^= 1
    elif grid[i-1][j]:
        res.append([i-1, j, i, j-1, i-1, j-1])
        grid[i][j-1] ^= 1
        grid[i-1][j-1] ^= 1
    grid[i][j] = grid[i-1][j] = 0
    return res

for _ in range(INT()):
    H, W = MAP()
    grid = [[]] * H
    for i in range(H):
        grid[i] = list(map(int, input()))

    ans = []
    if H % 2 == 1:
        # グリッド転置やれば縦横で別関数作らなくて済む
        grid = [list(row) for row in list(zip(*grid))]
        H, W = W, H
        # 関数内で添字を見やすくしたいので[0,H-1)じゃなくて[1,H)にする
        for i in range(1, H):
            res = check2(i, W-1)
            if res:
                for k in range(len(res)):
                    for l in range(0, 6, 2):
                        # 答えの行列入れ替え
                        res[k][l], res[k][l+1] = res[k][l+1], res[k][l]
                ans += res
        # 転置を元に戻す
        grid = [list(row) for row in list(zip(*grid))]
        H, W = W, H
    if W % 2 == 1:
        for i in range(1, H):
            ans += check2(i, W-1)
    for i in range(0, H-H%2, 2):
        for j in range(0, W-W%2, 2):
            ans += check1(i, j)

    print(len(ans))
    for row in ans:
        # グリッドと添字が一致してるほうが操作中見やすいので+1は最後にやる
        print(*[a+1 for a in row])
