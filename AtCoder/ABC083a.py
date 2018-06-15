# -*- coding: utf-8 -*-

A, B, C, D = map(int, input().split())

L = A + B
R = C + D

if L > R:
    print("Left")
elif L == R:
    print("Balanced")
elif L < R:
    print("Right")