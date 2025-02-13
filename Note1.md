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
与 DQN 中一样，采取类似于目标网络的方法，将上式中 $ r + \gamma V_{\omega}(s_{t+1}) $ 作为时序差分目标，不会产生梯度来更新价值函数

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

- Critic 负责估计 $V(s)$ ，但它的更新目标是 TD Target（即 $r_t + \gamma V(s_{t+1})$ ，等价于 $Q(s, a) $）。

- 这个 TD Target 作为“监督信号”，类似于监督学习中的 ground truth，Critic 通过均方误差最小化它。
- 也就是说，Critic 用 TD Target 作为监督信号 来让 $V(s)$ 逼近 $Q(s, a)$ 。

$$
L = \frac{1}{2} (V_{\theta}(s_t) - TD\ Target)^2 = \frac{1}{2} (V_{\theta}(s_t) - Q(s_t, a_t))^2
$$

这表明，Critic 其实是在用 TD 误差来优化 $V(s)$ ，让它逐步接近 $Q(s, a)$ 。



**Actor 和 Critic 的相互作用**

- Critic 估计 $V(s) $，帮助 Actor 评估策略的好坏。

- Actor 通过 TD 误差（优势估计）来优化策略，选择更好的动作，使得长期收益最大化。

- 当 Actor 变得更好时，Critic 也会有更好的 TD Target，从而让 Critic 估计的 $V(s)$ 更接近$ Q(s, a) $。



- 一开始，Critic 估计 $V(s)$ 可能不准确，因此 $Q(s, a)$ 和 $V(s)$ 可能差距较大。

- 但随着 Actor 选择更好的动作（即 最大化 $Q(s, a$) ）， $Q(s, a)$ 也会随着策略的改进而变化。

- 当训练趋于稳定，最优策略下，每个状态的动作都接近最优，导致 $Q(s, a)$ 和 $V(s)$ 差距变小，最终趋向于：

$$
Q(s, a) \approx V(s) \quad \text{（在最优策略下）}
$$

因为在最优策略下，每个状态的最优动作动作都能带来相同的价值。



## Chap11.TRPO



在策略梯度里，使用梯度上升来优化参数，使得策略 $\pi_{\theta}$ 最大化累积奖励。

但是，在深度强化学习中，策略通常由神经网络参数化，沿着梯度更新会导致策略的剧烈变化，可能会使得策略性能突然下降，甚至破坏已经学到的良好策略。



TRPO(Trust Region Policy Optimization) 就是为了解决这个问题，它引入了一个信任区域 trust region，确保每次更新策略的时候，策略的变化不会太大，从而保证学习的稳定性。



### 11.1 KL 散度

对于两个概率分布 $P(x)$ 和 $Q(x)$ ，它们的 **KL 散度** 定义为：
$$
D_{\text{KL}}(P || Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}
$$
或者在连续情况下：
$$
D_{\text{KL}}(P || Q) = \int P(x) \log \frac{P(x)}{Q(x)} dx
$$
其中：

- $P(x)$ 是真实分布（通常是目标分布）。
- $Q(x)$ 是近似分布（通常是当前模型的分布）。
- $\log \frac{P(x)}{Q(x)}$ 反映了在 $x$ 处， $Q(x)$ 偏离 $P(x)$ 的程度。
- **KL 散度是非对称的**，即 $D_{\text{KL}}(P || Q) \neq D_{\text{KL}}(Q || P) $。



**直观理解**：

- 当 $P(x)$ 和 $Q(x)$ 非常接近时， $D_{\text{KL}}(P || Q)$ 约等于 0。
- 如果 $Q(x)$ 在 $P(x)$ 可能性较大的地方概率很低（即 $Q(x)$ 没有很好地近似 $P(x)$ ），那么 KL 散度会很大。
- **KL 散度衡量的是 Q 相比于 P 的信息丢失**。



**KL散度可以看作是 P 分布下的 Q 分布的信息损失，或者说 P 期望的概率分布和 Q 之间的偏离程度。**

**衡量信息损失：**

- 若用 $Q(x)$ 来近似 $P(x)$，则 KL 散度衡量的是使用 $Q(x)$ 而损失了多少信息
- 例如，$P(x)$ 是真实世界的分布，而 $Q(x)$ 是神经网络的拟合分布，那么 KL 散度告诉的是 **模型还差多少**

$D_{\text{KL}}(P || Q) \neq D_{\text{KL}}(Q || P) $，它是基于 $P(x)$​ 计算的。



在 TRPO 和 PPO 中，KL 散度用于限制策略更新的幅度，防止策略突然崩坏，从而保证训练的稳定性。

KL 散度本质上1是一个衡量两个概率分布相似度的度量，帮助控制策略更新的稳定性。



### 11.2 重要性采样

重要性采样是一种统计方法，用于在不同的概率分布下进行样本估计。

核心思想是 **通过调整样本的权重** 来纠正因 **采样分布和目标分布不同** 所带来的偏差。



简单来说，**重要性采样**的目的是：

**从一个易于采样的分布（通常能容易地采样的分布）中生成样本，然后使用这些样本来估计另一个目标分布（关心的分布）的期望。**



在 RL 中，通常在某个策略下采样数据，但是实际关心的目标是另一个不同的策略下的数据的期望。

直接用一个策略的样本来估计另一个策略的期望会有偏差，**重要性采样通过给样本加权重来纠正这个偏差**。



假设有两个概率分布： $p(x)$ （可以采样的分布，行为分布）和 $q(x)$ （感兴趣的分布，叫做目标分布）。

计算目标分布下的期望：
$$
\mathbb{E}_{q}[f(x)] = \int f(x) q(x) dx
$$
但实际上无法从 $q(x)$ 采样，所以从 $p(x)$ 行为分布中采样，然后根据重要性定理来调整权重，使得在目标分布 $q(x)$ 下的期望更准确。



**公式：**
$$
\mathbb{E}_{q}[f(x)] = \int f(x) q(x) dx = \int f(x) \frac{q(x)}{p(x)} p(x) dx
$$
其中：

- $\frac{q(x)}{p(x)}$ 是重要性权重。它衡量了样本 $x$ 在目标分布 $q(x)$ 和行为分布 $p(x)$ 之间的差异。
- $p(x)$ 是从中采样的分布（行为分布）。
- $q(x)$ 是关心的分布（目标分布）。

通过从 **行为分布** $p(x)$ 中采样样本，然后根据 **权重** $\frac{q(x)}{p(x)}$ 来调整样本，从而正确估计目标分布 $q(x)$ 下的期望。



在RL中，策略优化的一个关键问题是如何通过采样旧策略的数据来优化新策略。因为我们有一个旧策略 $\pi_\theta$ ，但是要更新它并得到新策略 $\pi_{\theta{\prime}}$ 下的期望回报。用重要性采样解决。



直接上来理解：

- **行为分布**是从旧策略（行为分布）中采样数据，因为这些数据已经收集好了。
- **目标分布**是希望优化的是新策略（目标分布）。
- 调整权重由于新旧策略不同，通过重要性采样的比率来调整权重，从而让旧数据对新策略的估计更准确。





### 11.3 策略目标

首先，假设当前策略是 $\pi$ ，策略参数为 $\theta$ ，那么目标函数 $J(\theta)$ 的形式是期望 **从当前策略** $\pi_\theta$ **出发的回报**，即：
$$
J(\theta) = \mathbb{E}{\pi\theta} \left[ V^{\pi_\theta}(s_0) \right]\\

J(\theta) = \mathbb{E}{\pi\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$
由于初始状态 $s$​ 的分布于策略无关，所以：
$$
J(\theta) = \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t V^{\pi_\theta}(s_t) - \sum_{t=1}^\infty \gamma^t V^{\pi_\theta}(s_t) \right]
$$

$$
J(\theta) = \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t V^{\pi_\theta}(s_t) - \sum_{t=0}^\infty \gamma^{t+1} V^{\pi_\theta}(s_{t+1}) \right]
$$

$$
J(\theta) = -\mathbb{E}_{\pi_{\theta'}} \left[\sum_{t=0}^\infty \gamma^{t+1} V^{\pi_\theta}(s_{t+1}) -\sum_{t=0}^\infty \gamma^t V^{\pi_\theta}(s_t)\right]
$$

$$
J(\theta) = -\mathbb{E}_{\pi_{\theta'}} \left[\sum_{t=0}^\infty \gamma^t(\gamma V^{\pi_\theta}(s_{t+1}) -  V^{\pi_\theta}(s_t))\right]
$$

其中 $\gamma V(s_{t+1})-V(s_t)$ 是时序差分残差，新旧策略的目标函数的差距为：
$$
J(\theta') - J(\theta) = \mathbb{E}_{s_0} [V^{\pi_{\theta'}}(s_0)] - \mathbb{E}_{s_0} [V^{\pi_{\theta}}(s_0)] 
$$

$$
J(\theta') - J(\theta) = \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^{\infty} \gamma^t (r(s_t, a_t) + \gamma V^{\pi_{\theta}}(s_{t+1}) - V^{\pi_{\theta}}(s_t)) \right]
$$

其中，新策略的回报是 $\mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \right]$.



将时序差分残差定义为优势函数：
$$
A^{\pi_\theta}(s_t, a_t) = r(s_t, a_t) + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)
$$
因此，性能差距简化为：
$$
J(\theta') - J(\theta) = \mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t A^{\pi_\theta}(s_t, a_t) \right]
$$

- 优势函数 $A(s,a)$ 衡量在状态 $s$ 下选择动作 $a$ 的相对价值，相对于策略 $\pi_{\theta}$ 的值函数 $V(s)$
- 若新策略 $\pi_{\theta'}$ 在优势函数为正的动作上增加概率，则性能差距 $J(\theta')-J(\theta)$ 增大



在马尔可夫决策过程中，每个状态 $s_t$ 的访问概率可以用 **折扣状态访问分布** $\nu^{\pi_{\theta{\prime}}}(s)$ 表示：
$$
\nu^{\pi_{\theta{\prime}}}(s) = (1 - \gamma) \sum_{t=0}^{\infty} \gamma^t P(s_t = s | \pi_{\theta{\prime}})
$$
通过状态访问分布 $\nu^{\pi_{\theta{\prime}}}(s)$ ，可以将时间步上的累积求和转换为 **状态访问分布上的期望**：
$$
\mathbb{E}{\pi{\theta{\prime}}} \left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi_\theta}(s_t, a_t) \right] = \frac{1}{1 - \gamma} \mathbb{E}{s \sim \nu^{\pi{\theta{\prime}}}} \mathbb{E}{a \sim \pi{\theta{\prime}}} \left[ A^{\pi_\theta}(s, a) \right]
$$
这个公式的意思是：

新策略 $\pi_{\theta'}$ 是否比旧策略 $\pi_{\theta}$ 更好，取决于在所有访问的 s-a 下，优势函数的期望是否更大



直接优化这个公式很困难

**难点1：$\nu^{\pi_{\theta{\prime}}}(s)$ 依赖于 $\pi_{\theta'}$**

- 也就是说，需要用**新策略** $\pi_{\theta{\prime}}$ 来采样新的状态分布 $\nu^{\pi_{\theta{\prime}}}(s)$ ，但问题是现在还没有这个新策略
- 直觉上，想评价一个新方法，但还没试过它，怎么能知道它的效果？
- 这就相当于：想换工作，但是只有在换了之后才能知道新的工作好不好，这种情况很麻烦。

**难点2：遍历所有可能的新策略不现实**

- 若要尝试所有可能的 $\pi_{\theta'}$ 然后逐个计算回报，这不可行，策略空间太大了



TRPO（信任区域策略优化） 采用了一种近似的方法来简化计算：

1. **假设新策略** $\pi_{\theta{\prime}}$ **和旧策略** $\pi_{\theta}$ **差别不大**（即，每次更新策略的时候，不让它变化太大）。
2. **如果策略变化不大，那么状态访问分布** $\nu^{\pi_{\theta{\prime}}}(s)$ **也不会变太多**。
   - 也就是说，**新策略访问的状态分布和旧策略差不多**。
   - 所以我们可以**用旧策略的状态分布** $\nu^{\pi_{\theta}}(s)$ **来近似** $\nu^{\pi_{\theta{\prime}}}(s) $。
   - 这样就不需要重新采样新的数据了，而是直接用旧的数据来估计新策略的效果

于是，优化目标变成：
$$
L_{\theta}(\theta{\prime}) = J(\theta) + \frac{1}{1 - \gamma} \mathbb{E}{s \sim \nu^{\pi{\theta}}} \mathbb{E}{a \sim \pi{\theta{\prime}}} [ A^{\pi_{\theta}}(s, a) ]
$$
这个公式的意思是：

**我们用旧策略 $\pi_{\theta}$ 的状态访问分布 $\nu^{\pi_{\theta}}(s)$ 来计算新策略的期望，避免了直接计算 $\nu^{\pi_{\theta{\prime}}}(s) $的麻烦。**



**重要性采样修正**

状态分布用 $\pi_{\theta}$ 来近似，但是动作是由新策略 $\pi_{\theta'}$ 采样的，若直接用旧策略 $\pi_{\theta}$​ 采样的动作来估计，新策略的效果，可能会有偏差。



采用重要性采样来修正偏差，即：
$$
\mathbb{E}{\pi{\theta{\prime}}} [ A^{\pi_\theta}(s, a) ] = \mathbb{E}{\pi\theta} \left[ \frac{\pi_{\theta{\prime}}(a | s)}{\pi_\theta(a | s)} A^{\pi_\theta}(s, a) \right]
$$

$$
L_{\theta}(\theta{\prime}) = J(\theta) + \mathbb{E}{s \sim \nu^{\pi{\theta}}} \mathbb{E}{a \sim \pi{\theta}} \left[ \frac{\pi_{\theta{\prime}}(a | s)}{\pi_{\theta}(a | s)} A^{\pi_{\theta}}(s, a) \right]
$$



此外，还需要保证新旧策略足够接近，TRPO 用 KL 散步来衡量策略之间的距离，确保新旧策略状态访问分布变化很小。

TRPO 提出的优化目标是：
$$
\max_{\theta{\prime}} L_{\theta}(\theta{\prime}) \quad \text{subject to} \quad \mathbb{E}{s \sim \nu^{\pi{\theta}}} \left[ D_{\text{KL}} (\pi_{\theta_k}(\cdot | s), \pi_{\theta{\prime}}(\cdot | s)) \right] \leq \delta
$$
不等式约束定义了策略空间中的一个 KL 球，被称为信任区域。

在这个区域中，可以认为当前学习的策略和环境交互的状态分布于上一轮策略最后采样的状态分布一致。

可以基于一步行动的重要性采样方法使当前学习策略稳定提升。



### 11.4 近似求解

#### 11.4.1 多变量函数的泰勒展开

对于多变量函数 $f(\theta)$ ，在 $\theta_k$ 处的泰勒展开式为:
$$
f(\theta) \approx f(\theta_k) + g^T (\theta - \theta_k) + \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)
$$
其中：

- $g$ 是 **梯度**： $g = \nabla_{\theta} f(\theta)$ 表示函数在 $\theta_k$处的变化速率（方向和大小）。
- H 是 **黑塞矩阵**： **二阶导数** 的矩阵，表示函数在 $\theta_k$ 处的曲率，即函数的变化是如何加速或减速的。

$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial \theta_1^2} & \frac{\partial^2 f}{\partial \theta_1 \partial \theta_2} \\
\frac{\partial^2 f}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 f}{\partial \theta_2^2}
\end{bmatrix}
$$

在TRPO中，我们用黑塞矩阵来衡量 **KL 散度的曲率**，控制每次策略更新的步长，防止策略变化过快。

通过泰勒展开，能够在当前点附近做局部近似，利用梯度和曲率来决定更新策略的方向和幅度



#### 11.4.2 KKT 条件

在 TRPO 中，存在约束条件：
$$
\mathbb{E}{s \sim \nu{\pi_{\theta_k}}} \left[D_{\text{KL}}(\pi_{\theta_k}(\cdot | s), \pi_{\theta{\prime}}(\cdot | s))\right] \leq \delta
$$
希望找到一个新的策略参数 $\theta{\prime} $，使得目标函数 $L_{\theta}(\theta{\prime})$ 最大化。

同时，不能让新旧策略的 KL 散度变化太大，即 **策略更新不能过猛**，必须控制在 $\delta$ 以内。

这是一个带约束的优化问题。普通的梯度优化方法（比如梯度下降）无法直接解决这种问题，它们只适用于无约束优化。



KKT 条件是拉格朗日乘数法的推广，适用于约束的优化问题，可以用来找寻最优解。

假设要最大化一个函数：$\max_x f(x)$，同时存在一个约束 $g(x)\le0$

引入拉格朗日函数 $L(x,\lambda)=f(x)-\lambda g(x)$



**KKT条件的四个主要部分**

1. 可行性条件：$g(x) \le0$，最优解 $x$ 必须要满足约束条件
2. 拉格朗日乘子非负性：$\lambda^* \ge 0$，确保乘子有效，且能够约束目标函数
3. 互补松弛条件：$\lambda^* g(x^*) = 0$
   - **如果约束没有被“触及”（即** $g(x^*) < 0$ **）**，那么拉格朗日乘子 $\lambda^*$ = 0 ，即这个约束对优化目标没有影响
   - **如果约束正好被触及（即** $g(x^*) = 0$ **）**，那么拉格朗日乘子 $\lambda^*$​ > 0 ，说明约束在优化中起了作用。
4. KKT站立条件：

$$
\nabla f(x) - \lambda^ \nabla g(x) = 0
$$

最优解 $x$ 需要满足目标函数的梯度与约束函数的梯度的一个平衡，即方向要一致



**怎么在 TRPO 中应用 KKT 条件**

一个二次约束，即 KL 散度的约束，确保新旧策略之间的差异不超过 $\delta$
$$
\frac{1}{2} (\theta{\prime} - \theta_k)^T H (\theta{\prime} - \theta_k) \leq \delta
$$
为了求解这个带约束的优化问题，引入拉格朗日乘子 $\lambda$，构建拉格朗日函数：
$$
\mathcal{L}(\theta{\prime}, \lambda) = L_{\theta}(\theta{\prime}) - \lambda \left( \frac{1}{2} (\theta{\prime} - \theta_k)^T H (\theta{\prime} - \theta_k) - \delta \right)
$$
根据 **KKT 条件**，我们需要对 $\theta{\prime}$ 求导，得到以下条件：
$$
\nabla_{\theta{\prime}} L_{\theta}(\theta{\prime}) - \lambda H (\theta{\prime} - \theta_k) = 0
$$
给出了梯度更新方向，然后用 KKT 条件得到更新公式：
$$
\theta{\prime} = \theta_k + \lambda H^{-1} g
$$
其中：

- $g = \nabla_{\theta} L_{\theta}(\theta)$ 是目标函数的梯度。
- $H^{-1}$ 是黑塞矩阵的逆，表示 KL 散度的曲率。
- $\lambda$ 是拉格朗日乘子，控制约束对优化目标的影响。

接下来，通过 **互补松弛条件**，我们可以计算出 $\lambda$ ，从而得到最终的更新公式：
$$
\lambda = \sqrt{\frac{2\delta}{g^T H^{-1} g}}
$$
这确保了每次更新时KL 散度不会超过$ \delta$ ，保证了策略更新的稳定性。


$$
\theta_{k+1} = \theta_k + \sqrt{\frac{2\delta}{g^T H^{-1} g}} H^{-1} g
$$
其中：

- $g$ ：目标函数的梯度，决定了更新的方向。
- $H^{-1} g$ ：调整了更新的幅度，考虑了 KL 散度的曲率。
-  $\sqrt{\frac{2\delta}{g^T H^{-1} g}} $：确保每次更新都不会使 KL 散度超出阈值 $\delta$ ，防止更新过大。



### 11.5 共轭梯度

在神经网络中，策略函数有成千上万的参数。

若直接计算Hessian 矩阵的逆，所需的计算量和内存资源会非常庞大



TRPO 采用了共轭态度来解决这个问题，核心思想是：

直接计算梯度和黑塞矩阵的乘积，避免了直接计算Hessian 矩阵的逆

在 TRPO 中，假设 KL 散度的约束条件是：
$$
\frac{1}{2} (\beta x)^T H (\beta x) = \delta
$$
步长 $\beta$ 为：
$$
\beta = \sqrt{\frac{2\delta}{x^T H x}}
$$
这样确保了每次更新都不会导致 KL 散度超过阈值 $\delta$





## Chap12.PPO

PPO 基于 TRPO 的思想，有大量的实验结果表明，与 TRPO相比，PPO 能学的一样好，甚至更快



### 12.1 PPO-惩罚(Penalty)

PPO-惩罚（PPO-Penalty）用拉格朗日乘数法直接将 KL 散度的限制放进了目标函数中，变成了一个无约束的优化问题，在迭代的过程中不断更新 KL 散度前的系数。

核心思想是在优化目标中引入惩罚项，防止新策略与旧策略差异过大。



PPO-Penalty 的优化目标函数为：
$$
\arg \max_{\theta} \mathbb{E}{s \sim \nu{\pi_{\theta}}} \left[ \mathbb{E}{a \sim \pi{\theta}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a) - \beta D_{\text{KL}} \left( \pi_{\theta_k}(\cdot|s), \pi_{\theta}(\cdot|s) \right) \right] \right]
$$
其中：
- 第一项：基于重要性采样的策略梯度目标，通过新旧策略的概率比 $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ 加权优势函数 $A_t$，鼓励高优势动作
- 第二项：KL 散度惩罚项，用于约束新旧策略的差异，$\beta$ 为自适应惩罚系数。

#### 

为了动态调整惩罚系数 $\beta$，PPO 会根据 KL 散度 $d_k$ 的值动态调整惩罚系数 $\beta$。规则如下：

- **如果 KL 散度** $d_k$ **小于设定的阈值** $\delta / 1.5$ ，则减小 $\beta$ ：

$$
\beta_{k+1} = \frac{\beta_k}{2}
$$

​	这意味着，若 KL 散度较小，策略更新幅度也较小，可以进一步放松 KL 散度的约束，从而增大学习步长

- **如果 KL 散度** $d_k$ **大于** $\delta \times 1.5$ ，则增大 $\beta$ ：

$$
\beta_{k+1} = 2 \beta_k
$$

​	这意味着，若 KL 散度过大，表示策略更新幅度过大，需要加大惩罚，限制策略变化

- **否则**（即 $\delta / 1.5 \leq d_k \leq \delta$ ），不调整 $\beta$ ，保持当前值：

$$
\beta_{k+1} = \beta_k
$$

$\delta$ 为预设的超参数，用于控制 KL 散度的最大值，通常选在0.01 到 0.1 之间。



### 12.2 PPO-截断(Clip)

PPO-Clip 方法，直接在目标函数中进行限制，以确保新策略和旧策略的参数差距不会太大。

PPO-Clip 的优化目标是：
$$
\arg \max_{\theta} \mathbb{E}{s \sim \nu{\pi_{\theta_k}}} \mathbb{E}{a \sim \pi{\theta_k}(\cdot | s)}
\left[ \min \left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s, a),
\text{clip} \left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1 + \epsilon \right) A^{\pi_{\theta_k}}(s, a) \right) \right]
$$

- $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}$ 是新旧策略的概率比，衡量策略更新的幅度	
-  $\epsilon$ 用于限制策略更新的范围
- $clip(x, l, r) = max(min(x, r), l)$，即将 $x$ 限制在区间 $[l, r]$ 内



PPO-Clip 通过截断操作，限制策略的更新幅度，防止策略在单次更新时发生过大变化

- **如果** $A^{\pi_{\theta_k}}(s, a) > 0$，优化目标希望增大 $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}$ ，但不会超过 $1 + \epsilon$ 
- **如果** $A^{\pi_{\theta_k}}(s, a) < 0 $，优化目标希望减小 $\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}$ ，但不会低于 $1 - \epsilon$ 

这个限制防止策略在单次更新时发生过大变化，提高训练的稳定性，同时仍然允许一定程度的优化。







## Chap13. DDPG

Deep deterministic policy gradient 深度确定性策略梯度



DDPG 也属于 AC 算法，但是不同点在于，DDPG 是学习一个确定性策略，而不是像 REINFORCE 、TRPO 和 PPO 学习的是一个随机性策略。DDPG 的动作由函数直接确定，表示为 $α=μ_θ(s)$​。

不过虽然叫 PG，但是实际上应该是属于 DQN 的一个扩展版本。

### 13.1 离散动作与连续动作

对于连续动作，Q-learning 和 DQN 没办法处理。

对于离散动作，有几个动作就输出几个概率，$\pi_{\theta}(a_t \mid s_t)$ 随机性策略。

对于连续动作，比如要输出机械臂弯曲的角度，输出一个具体的浮点数 $\mu_{\theta}(s_t)$



对于确定性策略来说，输入某个状态 $s$ 输出的一定是相同的动作，而随机性策略还需要采样。



对于输出离散动作，加一个 softmax 确保所有的输出是动作概率

对于输出连续动作，一般可以在输出层 + tanh 函数。

其作用是，将输出限制到 $[-1,1]$，还可以乘上一个缩放因子，变到 $[-2,2]$ 等等。



### 13.2 DDPG 算法

DDPG 是 DQN 的扩展，可以扩展到连续动作空间。

在 DDPG 的训练中，它借鉴了DQN 的技巧：目标网络和经验回放。

经验回放与 DQN 是一样的，但目标网络的更新与DQN有点不同。



DDPG 在 DQN 基础上加了个策略网络来直接输出动作值，所以 DDPG 需要一边学 Q 网络，一边学习策略网络

Q 网络的参数用 $w$ 来表示。策略网络的参数用 $\theta$ 来表示。



最开始训练的时候，这两个神经网络的参数是随机的。

所以 critic 最开始是随机打分的，actor 也随机输出动作。但是由于有环境反馈的奖励存在，因此评论员的评分会越来越准确，所评判的演员的表现也会越来越好。

既然 actor 是一个神经网络，是我们希望训练好的策略网络，我们就需要计算梯度来更新优化它里面的参数 $\theta$。

简单来说，希望调整actor的网络参数，使得 critic 打分尽可能高。



**策略网络部分**

DQN 的最佳策略是想要学出一个很好的 Q 网络，学出这个网络之后，我们希望选取的那个动作使 Q 值最大。

DDPG 的目的也是求解让 Q 值最大的那个动作。

actor 只是为了迎合critic的打分而已，所以优化策略网络的梯度就是要最大化这个 Q 值，所以构造的损失函数就是让 Q 取一个负号。



**目标网络部分**

除了策略网络要做优化，DDPG 还有一个 Q 网络也要优化。

critic一开始也不知道怎么评分，它也是在一步一步的学习当中，慢慢地给出准确的分数。

优化 Q 网络的方法与 DQN 优化 Q 网络的方法是一样的，用真实的奖励 $r$ 和下一步的 $Q'$  拟合未来的奖励 $Q_{target}$

然后让 Q 网络的输出逼近 $Q_{target}$。



DDPG 需要用到4个神经网络，Actor 和 Critic 各用一个网络，此外各有一个目标网络。

在 DQN 中，每搁一段时间将 $Q$ 网络复制给目标 $Q$ 网络。

DDPG中，目标$Q$网络的更新，采取的是一种软更新的方式，让目标 $Q$ 网络缓慢更新，逐渐接近 $Q$ 网络：
$$
\omega^- \leftarrow \tau \omega + (1 - \tau) \omega^-
$$
 $\tau$ 等于1的时候，就和 DQN 的更新方式一样了。



DDPG 在行为策略上引入一个随机噪声 $N$ 来探索

![截屏2025-02-12 01.54.41](/Users/n/Library/Application Support/typora-user-images/截屏2025-02-12 01.54.41.png)





















