from typing import List

class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        N = len(mat)
        ans = 0
        for i in range(N):
            ans += mat[i][i]
        for i in range(N):
            ans += mat[i][N-i-1]
        # 奇数だと真ん中を重複して数えるので引く
        if N % 2 == 1:
            ans -= mat[N//2][N//2]
        return ans

# sol = Solution()
# print(sol.diagonalSum(mat))
