"""
・300点自力AC。スマホから。
・約数列挙
"""

def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]

def divisors(N: int) -> set:
    """ 約数の列挙・個数 """
    from math import sqrt

    # 1とその数はデフォで入れとく
    s = {1, N}
    # 終点はルート切り捨て
    for i in range(2, int(sqrt(N))+1):
        # 割り切れるなら、iとN/iを追加
        if N % i == 0:
            s.add(i)
            s.add(N // i)
    return s

N = INT()

ans = 10 ** 18
for div in divisors(N):
    ans = min(ans, div+N//div-2)
print(ans)
