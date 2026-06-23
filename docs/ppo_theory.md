# Theoretical Background: Proximal Policy Optimization (PPO)

This section documents the theoretical motivation and mathematical background of the PPO algorithm used in this Physical AI simulation.

---

## 1. Limitation of Vanilla Policy Gradient (VPG)

In Physical AI (continuous robot control), conventional Policy Gradient methods updating weights $\theta$ via standard Gradient Descent suffer from the **"Severe Policy Collapse"** issue.

* **Supervised Learning:** A wrong prediction updates the gradient based on a fixed dataset without altering future samples.
* **Reinforcement Learning:** A single sub-optimal update (e.g., excessive joint torque) alters the robot's physical state completely (e.g., falling over). This generates corrupted, highly biased trajectories, permanently stalling the learning loop.

Therefore, enforcing a **stable step size** that prevents catastrophic changes in the policy is critical for physical simulations.

---

## 2. From TRPO to PPO

To secure a stable update step, **TRPO (Trust Region Policy Optimization)** constrained the policy update using **KL-Divergence** as a hard constraint:

$$\mathbb{E} \left[ D_{KL}(\pi_{\theta_{old}}(\cdot|s) \parallel \pi_\theta(\cdot|s)) \right] \le \delta$$

While theoretically robust, TRPO requires computing the inverse of the **Fisher Information Matrix (FIM)**, leading to extreme computational complexity.
  * complexity : $O(N^2)$ or $O(N^3)$

**PPO (Proximal Policy Optimization)** replaces this heavy constraint with a simple **First-order Clipping Mechanism**, achieving comparable stability with higher computational efficiency.

---

## 3. Mathematical Objective Function

PPO optimizes the **Clipped Surrogate Objective Function** $L^{CLIP}(\theta)$:

$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$

### Key Components

#### 1) Probability Ratio 
$$r_t(\theta)$$
Tracks how much the new policy deviates from the old policy:
$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

#### 2) Advantage Estimate 
$$\hat{A}_t$$
Indicates whether the taken action performed better or worse than the baseline average.

#### 3) Clipped Mechanism 
$$\text{clip}$$
Limits the ratio $r_t(\theta)$ within the interval $[1-\epsilon, 1+\epsilon]$ (typically $\epsilon = 0.2$).
* **Positive Advantage ($\hat{A}_t > 0$):** The action yielded better performance. The policy increases the action probability, but the update is capped at $1+\epsilon$ to avoid sudden policy shifts.
* **Negative Advantage ($\hat{A}_t < 0$):** The action yielded worse performance. The update decreases the probability, bounded below by $1-\epsilon$ to mitigate policy collapse.

The $\min$ operator ensures a conservative bound, taking the unclipped or clipped value depending on which yields the lower (pessimistic) objective limit.

---

## 4. Key Metrics for Physical AI Debugging

When training in MuJoCo, the following logging metrics indicate policy stability:

| Metric | Interpretation | Physical Meaning |
| :--- | :--- | :--- |
| `ep_rew_mean` | Episode Reward Mean | Primary indicator of whether the robot is optimizing the task (e.g., forward velocity). |
| `clip_fraction` | Ratio of Clipped Data | The fraction of data points where $r_t(\theta)$ triggered the clip boundary. Reflects stable regulation of the trust region. |
| `approx_kl` | Approximate KL Divergence | Measures the statistical divergence between $\pi_{\theta_{old}}$ and $\pi_\theta$. High spikes signal unstable policy oscillation. |