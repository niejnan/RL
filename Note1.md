## Chap10.Actor-Critic

Actor-Critic 结合了 值函数 和 策略函数的思想：

Actor 决定怎么行动，根据状态 $s$ 选择动作 $a$

Critic 评价动作的好坏，指导 Actor 优化



AC 方法的核心思想是：

- **Actor 负责决策**：基于当前策略 policy $\pi(a | s)$选择动作。

- **Critic 负责评价**：使用 **价值函数** $V(s)$ 或 **优势函数** $A(s, a)$ 来衡量 Actor 选择的动作是否比平均水平更好

- **Actor 依赖 Critic 进行更新**：Critic 计算出的价值信号用于调整策略，使得高回报的动作被执行的概率更高



### 10.1 Actor-Critic

 在策略梯度中，可以把梯度写成这个更一般的形式：
$$
g = \mathbb{E} \left[ \sum_{t=0}^{T} \psi_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]
$$
其中，$\psi_t$可以有很多种形式： 

1. $\displaystyle \sum_{t'=0}^{T} \gamma^{t'} r_{t'}$：轨迹的总回报；
2. $\displaystyle \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$：动作$a_t$之后的回报； 
3. $\displaystyle \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} - b(s_t)$：基准线版本的改进； 
4. $Q^{\pi_\theta}(s_t, a_t)$：动作价值函数；
5. $A^{\pi_\theta}(s_t, a_t)$：优势函数； 
6. $r_t + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$​：时序差分残差。



> $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
>
> 优势估计，衡量动作 $a_t$ 的好坏，本质上就是 $A^{\pi}(s_t,a_t)=Q^{\pi}(s_t,a_t)-V^{\pi}(s_t)$​

虽然原始的 REINFORCE 对策略梯度的估计是无偏的，（采用蒙特卡洛），但是方差很大。

- 可以考虑引入一个 baseline function $b(s_t)$ 减小方差。

- 也可以采用 Actor-Critic 估计一个动作价值函数 $Q$，代替蒙特卡洛采样，即形式 $(4)$

- 也可以把状态价值函数 $V$ 当做 baseline，从 $Q$ 函数减去这个 $V$ 函数就得到了 $A$ 函数，也就是优势函数，即形式 $(5)$



这里主要讲形式6，也就是通过时序差分残差 $r_t + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$ 指导学习。

Actor 的更新用策略梯度完成， Critic 的更新用时序差分残差的学习方式，定义损失函数：
$$
\mathcal{L}(\omega) = \frac{1}{2} (r + \gamma V_{\omega}(s_{t+1}) - V_{\omega}(s_t))^2 
$$
与 DQN 中一样，我们采取类似于目标网络的方法，将上式中 $ r + \gamma V_{\omega}(s_{t+1}) $ 作为时序差分目标，不会产生梯度来更新价值函数。

因此，价值函数的梯度为：
$$
\nabla_{\omega} \mathcal{L}(\omega) = -(r + \gamma V_{\omega}(s_{t+1}) - V_{\omega}(s_t)) \nabla_{\omega} V_{\omega}(s_t)
$$
然后用梯度下降来更新就好了



### 9.2 使用 $r_t + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$ 

![截屏2025-02-07 20.53.55](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-07 20.53.55.png)

```python
for episode in range(num_episodes):  
    state = env.reset()  # 复位环境，获取初始状态
    
    while not done:
        # 1. Actor 选择动作 a
        action = actor.select_action(state)

        # 2. 执行动作，获取奖励 r 和下一个状态 s'
        next_state, reward, done, _ = env.step(action)

        # 3. 计算 TD 误差 δ
        td_error = reward + γ * critic.V(next_state) - critic.V(state)

        # 4. 更新 Critic（基于 TD 误差）
        critic_loss = td_error ** 2  # MSE 损失
        critic.optimizer.zero_grad()
        critic_loss.backward()
        critic.optimizer.step()

        # 5. 更新 Actor（策略梯度）
        log_prob = actor.get_log_prob(state, action)
        actor_loss = -log_prob * td_error  # 乘以 TD 误差作为优势
        actor.optimizer.zero_grad()
        actor_loss.backward()
        actor.optimizer.step()

        # 6. 进入下一个状态
        state = next_state
```



Update函数：

调用 update 函数时，是在一个 episode 完成后才进行的，所以这是 on-policy 方法，每一个 episode 里的 states = [s1,s2....] 的 s 个数不一定相同。

- 对于 actor 网络来说，输入 state，输出的是选择某个动作的概率。

- 对于 critic 网络来说，输入 state，输出的是 价值 $V(state)$

定义 $\delta_t$ 为：
$$
\delta_t = r_t +\gamma V^{\pi_\theta}(s_{t+1})-V^{\pi_{\theta}}(s_t)
$$

```python
td_target = rewards + gamma * critic(next_states) * (1 - done)
td_delta = td_target - critic(states)
```

td_target 就是 时序差分方法中，估计的真实状态价值

Critic(states) 是 critic 网络当当前状态的估计值 $V(s_t)$

Td Error = $r_t +\gamma V^{\pi_\theta}(s_{t+1})-V^{\pi_{\theta}}(s_t)$

Td 误差衡量的是当前 Critic 预测的 $V(s)$ 与 TD 目标之间的偏差



actor 输出概率，gather 选择当前的动作，log 对当前的概率取对数

```python
log_probs = torch.log(self.actor(states).gather(1, actions))
```



actor 的损失函数是：$\nabla J(\theta) \approx \log \pi(a_t|s_t) \cdot TD\ Error$

```
actor_loss = torch.mean(-log_probs * td_delta.detach())
```



> 为什么选择 Td 误差，而不是选择 $Q(s_t, a_t)$
>
> Td 误差本质上是对 Q 值的增量更新：
>
> - 直接用 $Q(s_t,a_t)$ 会受到 Critic 误差的影响，导致训练中数据的高方差
> - 使用 TD 误差作为优势估计，可以帮助策略更快收敛



```python
critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
```

使用均方误差
$$
L=\frac{1}{2}(V_{\theta}(s_t) - TD\ Target)^2=\frac{1}{2}(V_{\theta}(s_t) - (r_t+\gamma V(s_{t+1})))^2
$$
TD 目标本身就是 $r_t+\gamma V(s_{t+1})$，

这并不意味着是在学习动作价值函数 $Q(s,a)$，而是用来估计状态价值函数的 TD_target。

也就是说，优化的目标就是让 Critic 去学习，**从某个状态** $s_t$ **开始，按照当前策略继续执行下去所能获得的期望总回报**。

> 这个损失函数的意思是，Critic **通过最小化误差**（即 $V_{\theta}(s_t)$ 和 **TD Target** $r_t + \gamma V(s_{t+1})$ 之间的差异），来不断地调整它对状态 $s_t$ 的价值 $V(s_t)$ 的估计
>



类比与监督学习，在 **Actor-Critic** 里：

- **Critic 负责估计** $V(s)$ ，但它的更新目标是 **TD Target**（即 $r_t + \gamma V(s_{t+1})$ ，等价于 $Q(s, a) $）。

- 这个 **TD Target 作为“监督信号”**，类似于监督学习中的 **ground truth**，Critic 通过均方误差最小化它。
- 也就是说，Critic **用 TD Target 作为监督信号** 来让 $V(s)$ 逼近 $Q(s, a)$ 。

$$
L = \frac{1}{2} (V_{\theta}(s_t) - TD\ Target)^2 = \frac{1}{2} (V_{\theta}(s_t) - Q(s_t, a_t))^2
$$

这表明，**Critic 其实是在用 TD 误差来优化** $V(s)$ **，让它逐步接近** $Q(s, a)$ 。



**Actor 和 Critic 的相互作用**

- **Critic 估计** $V(s) $，帮助 Actor 评估策略的好坏。

- **Actor 通过 TD 误差（优势估计）来优化策略**，选择更好的动作，使得长期收益最大化。

- **当 Actor 变得更好时，Critic 也会有更好的 TD Target，从而让 Critic 估计的** $V(s)$ **更接近**$ Q(s, a) $。



- 一开始，Critic 估计 $V(s)$ 可能不准确，因此 $Q(s, a)$ 和 $V(s)$ 可能差距较大。

- 但随着 Actor 选择更好的动作（即 **最大化** $Q(s, a$) ）， $Q(s, a)$ 也会随着策略的改进而变化。

- 当训练趋于稳定，**最优策略下，每个状态的动作都接近最优，导致** $Q(s, a)$ **和** $V(s)$ **差距变小**，最终趋向于：

$$
Q(s, a) \approx V(s) \quad \text{（在最优策略下）}
$$

因为在最优策略下，每个状态的最优动作动作都能带来相同的价值。

















