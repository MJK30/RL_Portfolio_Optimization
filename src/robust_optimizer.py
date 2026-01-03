import cvxpy as cp
import numpy as np

def robustify_ppo_weights(ppo_weights, mu_hat, cov_matrix, vix_today, lambda_risk=2.0, delta=0.3):
    """
    Adjusts PPO weights for uncertainty using a Box Uncertainty Set.
    ppo_weights: Raw weights from the RL agent.
    mu_hat: 30-day rolling average returns (point estimates).
    cov_matrix: Current covariance matrix of assets (risk model).
    vix_today: Current VIX level to scale the 'Safety Range'.
    """
    
    n = len(ppo_weights)
    w = cp.Variable(n)
    
    # Defining the Safety Range based on VIX
    # High volatility (high VIX) -> larger uncertainty -> wider box
    # Low volatility (low VIX) -> smaller uncertainty -> narrower box
    epsilon = delta * (vix_today / 100)  # Scale delta by VIX level
    
    
    # Objective function
    # Term 1: Portfolio Risk -> Variance
    risk = cp.quad_form(w, cov_matrix)
    
    # Term 2: Worst-case Return under Box Uncertainty
    # Penalty is higher if you concentrate too much in one asset (cp.norm(w, 1))
    worst_case_return = cp.sum(cp.multiply(w, mu_hat)) - epsilon * cp.norm(w, 1)
    
    
    # Full Objective: Find weights that minimize Risk - (Weight * Worst_Case_Return)
    objective = cp.Minimize(risk - lambda_risk * worst_case_return)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,                # Constraint 1: Fully invested
        w >= 0,                       # Constraint 2: No short selling
        cp.norm(w - ppo_weights, 1) <= delta  # Constraint 3: Trust Region (Max 50% shift from PPO)
    ]
    
    # Solving the problem
    prob = cp.Problem(objective, constraints)
    try:
        # We use OSQP because it is optimized for quadratic problems
        prob.solve(solver=cp.OSQP)
        if w.value is None: return ppo_weights # Fallback to PPO if solver fails
        return w.value
    except:
        return ppo_weights # Safety fallback