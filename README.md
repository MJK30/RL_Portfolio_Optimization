# RL-Based Multi-Asset Portfolio Optimization

This project implements a hybrid financial investment strategy that combines Reinforcement Learning (RL) for alpha generation with a Robust Optimization layer for risk control. It is designed to outperform a standard Equal-Weight (EW) benchmark by dynamically adjusting to market volatility and uncertainty.

## Core Concepts

### 1. The Proximal Policy Optimization Agent (Alpha Generation)
The strategy utilizes Proximal Policy Optimization (PPO), a state-of-the-art Reinforcement Learning algorithm.

- Goal: To learn a policy that maximizes long-term portfolio returns based on technical indicators and macroeconomic data.

- Observation Space: Includes technical indicators (SMA, RSI, MACD), volatility (VIX), and macroeconomic data (GDP, CPI, Unemployment).

- Output: The agent provides "Raw Weights" for the asset universe, which currently includes GLD, SPY, and USO.

### Robust Optimization (Risk Manager)


Because RL agents can be overly aggressive or "overfit" to specific market conditions, the system includes a Robust Optimization Layer.

- Box Uncertainty: The model assumes return estimates ($\hat{\mu}$) are not perfect and instead lie within a "safety range" scaled by the VIX level.

- Worst-Case Minimization: The optimizer seeks weights that minimize portfolio variance (risk) while maximizing the "worst-case" possible return within that uncertainty range.

- Trust Region: A constraint ensures the final robust weights do not drift more than a specific percentage (delta) from the RL agent's original suggestion.


## Files Architecture & Algorithms

### src/environment.py

- Role: Custom OpenAI Gym/Gymnasium environment.

- Logic
    - Manages the State Space, feeding the PPO agent the necessary market context.
    - Implements the Step Function, which applies weights to asset returns to calculate the next portfolio value.
    - Returns rewards based on portfolio performance.

### src/robust_optimizer.py

- Algorithm: Convex Optimization using the OSQP solver via CVXPY.

- Key Features:
    - Covariance Matrix: Represents the risk model of the assets.
    - Penalty Term: A penalty is applied for high concentration in specific assets to encourage diversification.
    - Fallbacks: Includes a safety mechanism to return raw PPO weights if the solver fails to find an optimal solution.   

### src/evaluate.py

- Role: The backtesting and reporting engine.

- Backtest Logic:
    - Iterates through historical data starting from 2020.
    - Calculates 30-day rolling Covariance Matrices and mean returns for the optimizer.

- Benchmark: Compares performance against an Equal-Weight (EW) Strategy (1/3 GLD, 1/3 SPY, 1/3 USO).

## Backtest Results

Based on historical simulations, the hybrid PPO-Robust strategy shows competitive performance against a balanced benchmark.

Strategy,Annual Return,Sharpe Ratio,Max Drawdown
PPO Agent (Robust),19.03%,0.909,25.59%
Equal-Weight,18.66%,1.198,18.40%



# References

@book{cornuejols2006optimization,
  title={Optimization Methods in Finance},
  author={Cornu{\'e}jols, G{\'e}rard and T{\"u}t{\"u}nc{\"u}, Reha},
  year={2006},
  month={January},
  publisher={Carnegie Mellon University},
  address={Pittsburgh, PA}
}


@mastersthesis{ganesh2025reinforcement,
  title={Reinforcement Learning Meets Robust Portfolio Optimization: A Dynamic Approach to Risk-Aware Asset Allocation},
  author={Ganesh, Nimmagadda Sai Bhanu Prasanna},
  year={2025},
  month={April},
  school={RWTH Aachen University},
  type={Master Thesis},
  address={Aachen, Germany},
  note={1st Examiner: Dr. sc. pol. Thomas Lontzek; 2nd Examiner: Dr. rer. pol. Sven M{\"u}ller}
}

