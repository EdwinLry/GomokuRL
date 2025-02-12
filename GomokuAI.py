import random
from torch import nn, optim
import torch


class GomokuAI(nn.Module):
    def __init__(self, board_size=15, player=2):
        super(GomokuAI, self).__init__()
        # Define the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.board_size = board_size
        # Build and move the model to the device
        self.model = self.build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = []
        self.player = player

    def build_model(self):
        return nn.Sequential(
            nn.Linear(self.board_size * self.board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.board_size * self.board_size)
        )

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.board_size * self.board_size - 1)
        # Convert state to a tensor and move it to the device
        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def compute_reward(self, winner, current_player, valid_move=True, game_state=None, move=None):
        if not valid_move:
            return -5  # Penalty for invalid moves

        if winner == current_player:
            return 1  # Reward for winning

        if winner == 3 - current_player:
            return -1  # Penalty for losing

        return 0  # Default neutral move

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Create tensors and move them to the device
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state.flatten()).unsqueeze(0).to(self.device)

            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            # Clone current Q-values and update the target for the taken action
            target_f = self.model(state_tensor).clone()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename="gomoku_ai.pth"):
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, filename)
        print("Model saved.")

    def load(self, filename="gomoku_ai.pth"):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        # Move the model to the device after loading
        self.model.to(self.device)
        print("Model loaded.")
