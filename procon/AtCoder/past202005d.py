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
INF = 10 ** 19
MOD = 10 ** 9 + 7

nums = [''] * 5
nums[0] = '.###..#..###.###.#.#.###.###.###.###.###'
nums[1] = '.#.#.##....#...#.#.#.#...#.....#.#.#.#.#'
nums[2] = '.#.#..#..###.###.###.###.###...#.###.###'
nums[3] = '.#.#..#..#.....#...#...#.#.#...#.#.#...#'
nums[4] = '.###.###.###.###...#.###.###...#.###.###'

N = INT()
grid = [''] * 5
for i in range(5):
    grid[i] = input()

def check(x):
    for k in range(10):
        ok = True
        for i in range(5):
            for j in range(4):
                if nums[i][4*k+j] != grid[i][4*x+j]:
                    ok = False
        if ok:
            return k

ans = []
for i in range(N):
    ans.append(check(i))
print(''.join(map(str, ans)))
