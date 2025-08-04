"""
Maximum Students Taking Exam
Given a 2D array representing seats in a classroom ('.' for empty, '#' for broken), find the maximum number of students that can take the exam without any two students sitting next to each other (left/right) or diagonally (upper left/right).
"""

class Solution:
    def maxStudents(self, seats):
        m, n = len(seats), len(seats[0])  # get the number of rows and columns
        dp = [{} for _ in range(m + 1)]  # dp[row][mask] = max students up to this row with this mask
        dp[0][0] = 0  # start with no students seated
        
        # helper to check if a mask is valid for a row
        def is_valid(row, mask):  # helper to check if a mask is valid for a row
            for j in range(n):  # check each seat
                if (mask & (1 << j)) and seats[row][j] == '#':  # can't sit on a broken seat
                    return False
                if (mask & (1 << j)) and (j > 0 and (mask & (1 << (j - 1)))):  # can't sit next to another student
                    return False
            return True
        
        for row in range(1, m + 1):  # loop through each row
            for prev_mask in dp[row - 1]:  # loop through all previous masks
                for curr_mask in range(1 << n):  # try all possible seatings for this row
                    if is_valid(row - 1, curr_mask):  # check if current mask is valid
                        # check for diagonal adjacency with previous row
                        if (curr_mask & (prev_mask << 1)) == 0 and (curr_mask & (prev_mask >> 1)) == 0:  # check for diagonal adjacency
                            count = bin(curr_mask).count('1')  # count the number of students in this mask
                            dp[row][curr_mask] = max(dp[row].get(curr_mask, 0), dp[row - 1][prev_mask] + count)  # update max students
        return max(dp[m].values())  # return the maximum number of students 