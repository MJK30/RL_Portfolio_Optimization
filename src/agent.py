import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import PortfolioEnv



def train_agent():
    
    # Load the preprocessed data
    df = pd.read_csv('data/processed_features.csv', index_col=0)
    
    # Initialize the custom environment
    # DummyVecEnv is used to vectorize the environment for Stable Baselines3
    env = DummyVecEnv([lambda: PortfolioEnv(df)])
    
    # Initialize the PPO agent
    # PPO is Actor-Critic method suitable for continuous action spaces
    # MlpPolicy indicates a feedforward neural network policy which builds the Policy and Value networks
    # Proximal Clipping is used to ensure stable updates during training
    # Discount factor (gamma) determines the importance of future rewards when the values are computed
    model = PPO(
        "MlpPolicy", # MLP Architecture
        env,
        verbose=1, 
        learning_rate=0.0003, # Speed of weight updates
        gamma=0.99, # Discount factor for future rewards
        clip_range=0.2,
        n_steps=2048, # Number of steps to run for each environment per update
        batch_size=64, # Minibatch size for each gradient update
    )
    
    print("Start Training!")
    model.learn(total_timesteps=100000) # Train for 100,000 timesteps
    
    model.save("models/ppo_portfolio_model")
    print("Model saved as 'models/ppo_portfolio_model'")
    

if __name__=="__main__":
    train_agent()