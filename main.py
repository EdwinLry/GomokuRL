import random

import torch

import gamelogic
import GomokuAI
from GomokuAI import GomokuAI


def run_game():
    ai = GomokuAI()
    game = gamelogic.Gomoku(current_player=random.randint(1, 2))
    game.reset()
    print(game.board)
    while True:
        if game.is_full():
            print("Tie.")
            break
        if game.current_player == 1:
            action = ai.select_action(game.board)
            x, y = divmod(action, ai.board_size)
            # player1 inputs
        else:
            action = ai.select_action(game.board)
            x, y = divmod(action, ai.board_size)
            # player2 inputs
        board, winner = game.make_move(x, y)
        print(board)
        print("\n")
        if winner:
            print(f"Player {winner} wins!")
            break


def train_ai(num_episodes=1000, save_interval=50, train_every=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ai = GomokuAI().to(device)
    ai.load()
    ai.save(filename="previous_ai.pth")
    for episode in range(num_episodes):
        game = gamelogic.Gomoku()
        game.reset()
        done = False
        step_counter = 0
        invalid_moves = 0
        while not done:
            state = game.board
            action = ai.select_action(state)
            x, y = divmod(action, ai.board_size)
            # Optionally print only every N steps or for debug episodes
            next_state, winner = game.make_move(x, y)
            if next_state is None:
                reward = -5  # Strong penalty for invalid moves
                next_state = state.copy()  # Prevent storing None
                ai.store_experience(state, action, reward, next_state, done)
                invalid_moves += 1
            else:
                reward = ai.compute_reward(winner, game.current_player)
                ai.store_experience(state, action, reward, next_state, done)

            step_counter += 1
            # Train every train_every step (batch training)
            if step_counter % train_every == 0:
                ai.train()

            # Optionally, update 'done' based on game logic
            if game.is_full():
                done = True

            if winner:  # or some other terminal condition
                done = True

        # Final training at end of episode (if any experiences are left)
        ai.train()
        print(f"Episode {episode} finished after {step_counter} steps. With {invalid_moves} invalid moves.")
        if episode % save_interval == 0:
            ai.save()
            print(f"Episode {episode}:\n {game.board}\n")

    ai.save()


def evaluate_agent(num_games=100):
    wins = 0
    current_agent = GomokuAI()
    current_agent.load()
    opponent_agent = GomokuAI()
    opponent_agent.load(filename="previous_ai.pth")
    for i in range(num_games):
        game = gamelogic.Gomoku(current_player=random.randint(1, 2))
        game.reset()
        state = game.board
        done = False
        while not done:
            if game.current_player == 1:
                action = current_agent.select_action(state)
                print(1)
            else:
                action = opponent_agent.select_action(state)
                print(2)
            x, y = divmod(action, current_agent.board_size)
            print(f"Player {game.current_player} moves: ({x}, {y})")
            next_state, winner = game.make_move(x, y)
            if next_state is None:
                print("Invalid move.")
                done = True
            state = next_state
            if winner is not None:
                done = True
                if winner == 2:
                    wins += 1
            if game.is_full():
                print("Tie.")
                done = True
        print(f"{game.board}\n")
    return wins / num_games

def main():
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    #evaluate_agent(1)
    train_ai()
    #run_game()


if __name__ == "__main__":
    main()
