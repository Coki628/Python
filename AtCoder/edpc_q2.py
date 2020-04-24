# -*- coding: utf-8 -*-

"""
・なんかpythonももうちょいで通りそうだったから通しといた。
・やっぱクラスで持ってたやつ関数にすると0.1~0.2秒くらいは速くなるな。
"""

def main():
    import sys

    def input(): return sys.stdin.readline().strip()
    def INT(): return int(input())
    def MAP(): return map(int, input().split())
    def LIST(N=None): return list(MAP()) if N is None else [INT() for i in range(N)]
    
    def query(i):
        s = intv
        i += 1
        while i > 0:
            s = max(s, tree[i-1])
            i -= i & -i
        return s
    
    def update(i, x):
        i += 1
        while i <= size:
            tree[i-1] = max(tree[i-1], x)
            i += i & -i

    n = INT()
    H = LIST()
    A = LIST()

    n += 2
    nv = 1
    while nv < n:
        nv *= 2
    size = nv
    intv = 0
    tree = [intv] * nv

    for h, a in zip(H, A):
        mx = query(h)
        update(h, mx + a)
    ans = query(n)
    print(ans)
main()
