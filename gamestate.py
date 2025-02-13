import numpy as np


def check_open(board, move, num) -> bool:
    x, y = move
    player = board[x, y]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal, Anti-diagonal

    def validate(mx, my):
        return 0 <= mx < board.shape[0] and 0 <= my < board.shape[1]

    for dx, dy in directions:
        count = 1
        for i in range(1, num + 1):  # Check in one direction
            nx, ny = x + i * dx, y + i * dy
            if (nx == 0 and x != 0) or (ny == 0 and y != 0):
                break
            if validate(nx, ny) and board[nx, ny] == player:
                count += 1
            else:
                if validate(nx, ny) and board[nx, ny] != 0:
                    return False
                break

        for i in range(1, num + 1):  # Check in the opposite direction
            nx, ny = x - i * dx, y - i * dy
            if (nx == 0 and x != 0) or (ny == 0 and y != 0):
                break
            if validate(nx, ny) and board[nx, ny] == player:
                count += 1
            else:
                if validate(nx, ny) and board[nx, ny] != 0:
                    return False
                break

        if count == num:
            return True
    return False


def blocking(board, move):
    """Methods rewards blocking a 3 or 4"""
    pass


def multi_threats(board, move):
    """Massive reward for making multiple threats(open 3's one sided 4) simultaneously """
    pass
