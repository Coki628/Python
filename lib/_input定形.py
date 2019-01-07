# -*- coding: utf-8 -*-

# 高速な標準入力
import sys
def input(): return sys.stdin.readline().strip()

# 数値型
N = int(input())
# 数値型で各変数
A, B, C = map(int, input().split())
# 数値型リスト
aN = list(map(int, input().split()))
# 数値型リスト(どっちでも一緒だと思う)
aN = [int(i) for i in input().split()]
# 数値型リスト複数行
[list(map(int, input().split())) for i in range(N)]
# 連続した数値を1文字ずつ各変数
A, B, C = map(int, list(input()))
# 複数行の数値を格納したリスト
aN = [int(input()) for i in range(N)]

# 文字列型
s = input()
# 文字列型リスト
sN = input().split()
# 1文字ずつ文字列型リスト
sN = list(input())
# 複数行の文字列を格納したリスト
sN = [input() for i in range(N)]

# 縦*横の2次元配列
H, W = map(int, input().split())
# 全0の初期化
HW = [[0] * W for i in range(H)]
HW = np.zeros((H,W,))
# N個の空配列で初期化
[[] for i in range(N)]
# 入力から値を読み込む(スペース無文字列)
HW = []
for i in range(H):
    HW.append(list(input()))
# 入力から値を読み込む(スペース有数値)
HW = []
for i in range(H):
    HW.append(list(map(int, input().split())))
# 入力から値を読み込む(リスト内包表記)
HW = [list(input()) for i in range(H)]
cN = [list(map(int, input().split())) for i in range(N)]
# 8方向を見に行く用の配列を準備しておく
dx = [1, 0, -1, 0, 1, -1, -1, 1]
dy = [0, 1, 0, -1, 1, 1, -1, -1]
# 結果出力
for i in range(H):
    print("".join(HW[i]))

# xとyを別配列にそれぞれ保持
xN, yN = [0]*N, [0]*N
for i in range(N):
    xN[i], yN[i] = map(int, input().split())
# cとtをペアにして2次元配列に保持
cNtN = [[0] * 2 for i in range(N)]
for i in range(N):
    cNtN[i][0], cNtN[i][1] = map(int, input().split())
# N*～の空の2次元配列
xNyN = [[] for i in range(N)]
