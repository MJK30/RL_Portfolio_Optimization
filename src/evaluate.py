import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import PortfolioEnv
from robust_optimizer import robustify_ppo_weights


def calculate_metrics(values):
    """Calculates key financial performance metrics."""
    returns = pd.Series(values).pct_change().dropna()
    
    # 1. Annual Return 
    total_return = (values[-1] / values[0]) - 1
    annual_return = ((1 + total_return) ** (252 / len(values))) - 1
    
    # 2. Sharpe Ratio
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252)
    
    # 3. Maximum Drawdown 
    peak = pd.Series(values).cummax()
    drawdown = (peak - values) / peak
    max_drawdown = drawdown.max()
    
    return annual_return, sharpe_ratio, max_drawdown

def run_backtest():
    # Load data and model
    df = pd.read_csv("data/processed_features.csv", index_col=0)
    model = PPO.load("models/ppo_portfolio_model")
    
    # Initialize environment
    env = PortfolioEnv(df)
    obs, _ = env.reset()
    
    ppo_values = [1.0]
    ew_values = [1.0]
    
    print("Running backtest simulation...")
    done = False
    asset_return_cols = ["GLD_Return", "SPY_Return", "USO_Return"]
    while not done:
        # A. Get PPO Weights 
        action, _ = model.predict(obs, deterministic=True)
        raw_ppo_weights = action / (np.sum(action) + 1e-10)
        
        # 2. Extract context from the environment for the Optimizer
        # We need the last 30 days of returns to calculate the Risk (Covariance)
        current_idx = env.current_step
        
        if current_idx >= 30:            
            
            lookback_slice = df[asset_return_cols].iloc[current_idx-30 : current_idx]
            mu_hat = lookback_slice.mean().values           # Average return estimate
            cov_matrix = lookback_slice.cov().values        # The risk model
            cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Force symmetry
            vix_today = df.iloc[current_idx]['VIX']         # Current market fear level

            # Apply the Robust Layer 
            robust_weights = robustify_ppo_weights(
                ppo_weights=raw_ppo_weights,
                mu_hat=mu_hat,
                cov_matrix=cov_matrix,
                vix_today=vix_today
            )
        else:
            # Use raw PPO weights until we hit day 30
            robust_weights = raw_ppo_weights
        
        obs, reward, done, truncated, info = env.step(robust_weights)
        ppo_values.append(info['portfolio_value'])
        
        # B. Get Equal-Weight Values (1/3 for each asset) [cite: 1936]
        # Calculate EW return manually for the same day
        asset_returns = [df.iloc[env.current_step-1][col] for col in asset_return_cols]
        ew_return = np.mean(asset_returns)
        ew_values.append(ew_values[-1] * (1 + ew_return))

    # C. Evaluation and Reporting [cite: 1933]
    ppo_metrics = calculate_metrics(ppo_values)
    ew_metrics = calculate_metrics(ew_values)
    
    print(f"\n--- Backtest Results ---")
    print(f"Strategy      | Annual Return | Sharpe Ratio | Max Drawdown")
    print(f"PPO Agent     | {ppo_metrics[0]:.2%}       | {ppo_metrics[1]:.3f}      | {ppo_metrics[2]:.2%}")
    print(f"Equal-Weight  | {ew_metrics[0]:.2%}       | {ew_metrics[1]:.3f}      | {ew_metrics[2]:.2%}")

    # D. Plotting the results [cite: 1954]
    plt.figure(figsize=(12, 6))
    plt.plot(ppo_values, label='PPO Strategy', linewidth=2)
    plt.plot(ew_values, label='Equal-Weight Strategy', linestyle='--', alpha=0.7)
    plt.title("Portfolio Value Evolution: PPO vs Equal-Weight")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.savefig("data/backtest_results.png")
    plt.show()

if __name__ == "__main__":
    run_backtest()