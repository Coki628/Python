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
EPS = 10 ** -10

X = INT()

if 400 <= X < 600:
    print(8)
elif 600 <= X < 800:
    print(7)
elif 800 <= X < 1000:
    print(6)
elif 1000 <= X < 1200:
    print(5)
elif 1200 <= X < 1400:
    print(4)
elif 1400 <= X < 1600:
    print(3)
elif 1600 <= X < 1800:
    print(2)
else:
    print(1)
