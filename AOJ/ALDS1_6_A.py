# -*- coding: utf-8 -*-

"""
バケットソート
・累積和
"""

def INT(): return int(input())
def MAP(): return map(int, input().split())
def LIST(): return list(map(int, input().split()))

N = INT()
A = LIST()

def CountingSort(A, k):
    C = [0] * (k+1)
    B = [0] * N
    # C[i] に i の出現数を記録する
    for i in range(N):
        C[A[i]] += 1
    # C[i] に i 以下の数の出現数を記録する
    for i in range(1, k+1):
        C[i] += C[i-1]

    for i in range(N-1, -1, -1):
        B[C[A[i]]-1] = A[i]
        # C[A[i]]の値は1つ場所を確定させたので、1減らしておく
        C[A[i]] -= 1
    return B

B = CountingSort(A, max(A))
print(*B)
