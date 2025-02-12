import numpy as np


class Gomoku:
    def __init__(self, board_size=15, current_player=1):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
        self.current_player = current_player  # Player 1 starts

    def reset(self):
        """Reset the board to start a new game."""
        self.board.fill(0)
        self.current_player = 1
        return self.board

    def is_valid_move(self, x, y):
        """Check if a move is valid."""
        return 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x, y] == 0

    def make_move(self, x, y):
        """Place a stone on the board."""
        if self.is_valid_move(x, y):
            self.board[x, y] = self.current_player
            winner = self.check_winner(x, y)
            self.current_player = 3 - self.current_player  # Switch players (1 -> 2, 2 -> 1)
            return self.board, winner
        else:
            return None, None  # Invalid move

    def check_winner(self, x, y):
        """Check if the player wins after placing a stone at (x, y)."""
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal, Anti-diagonal
        player = self.board[x, y]

        for dx, dy in directions:
            count = 1
            for i in range(1, 5):  # Check in one direction
                nx, ny = x + i * dx, y + i * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break

            for i in range(1, 5):  # Check in the opposite direction
                nx, ny = x - i * dx, y - i * dy
                if 0 <= nx < self.board_size and 0 <= ny < self.board_size and self.board[nx, ny] == player:
                    count += 1
                else:
                    break

            if count >= 5:
                return player  # Winner found

        return 0  # No winner yet

    def is_full(self):
        """Check if the board is full."""
        return not (self.board == 0).any()

