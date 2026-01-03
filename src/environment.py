import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd



class PortfolioEnv(gym.Env):
    """
    Custom Environment for Financial Portfolio Management
    Simulates a trading environment where an agent can manage a portfolio of assets.
    Simulates daily trading of SPY, GLD and USO
    """
    
    def __init__(self, df):
        super(PortfolioEnv, self).__init__()
        self.df = df
        self.tickers = ["SPY", "GLD", "USO"]
        self.n_assets = len(self.tickers)
        
        # Action Space: Continuous weights [0,1] for each asset
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        
        # Observation Space: The current state of the market (21+ features)
        # This is the State representation for the Agent in the environment
        # Each State, is a vector of all features for all assets at the current timestep
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.df.shape[1],), dtype=np.float32)
        
        self.current_step = 0
        self.total_steps = len(self.df) - 1
        self.portfolio_value = 1.0 # Normalized initial portfolio value
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the first day of the dataset

        Args:
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = 1.0
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs, {}
    
    def step(self, action):
        """
        Execute one day of tradding based on the agents weights

        Args:
            action (_type_): _description_
        """
        
        # Normalization
        weights = action / (np.sum(action) + 1e-8)
        
        # Reward Calculation: Dot product of weights and daily returns
        # For each Observation (State) - Action -> Reward is computed
        asset_returns = np.array([self.df.iloc[self.current_step][f"{tick}_Return"] for tick in self.tickers])
        portfolio_return = np.dot(weights, asset_returns)
        
        # update the portfolio value: V_{t+1} = V_t * (1 + r_t)
        self.portfolio_value *= (1 + portfolio_return)
        
        
        # Advance to next step
        self.current_step += 1
        done = self.current_step >= self.total_steps
        truncated = False
        
        # define the observation for the next step
        if not done:
            obs = self.df.iloc[self.current_step].values.astype(np.float32)
        else:
            obs = np.zeros(self.observation_space.shape)
            
        info = {"portfolio_value": self.portfolio_value, "weights": weights}
        return obs, portfolio_return, done, truncated, info
        
        
        
