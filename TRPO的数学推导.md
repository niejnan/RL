## TRPO的数学推导

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



#### 目标等式推导

TRPO的目标等式为：
$$
\mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^{\infty} \gamma^t A^{\pi_\theta}(s_t, a_t) \right] = \frac{1}{1 - \gamma} \mathbb{E}_{s \sim \nu^{\pi_{\theta'}}} \mathbb{E}_{a \sim \pi_{\theta'}} \left[ A^{\pi_\theta}(s, a) \right]
$$
**1.展开左侧的期望** 

这个等式的左侧，是轨迹的期望，可以分解为每个时间步的期望之和，即：
$$
\mathbb{E}_{\pi_{\theta'}} \left[ \sum_{t=0}^\infty \gamma^t A^{\pi_\theta}(s_t, a_t) \right] = \sum_{t=0}^\infty \gamma^t \mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ A^{\pi_\theta}(s_t, a_t) \right]
$$
这一步利用了期望的线性性，即线性叠加性，将求和符号移动到了期望以外。



**2.分解时间步的期望**

对于每个时间步 $t$，期望可以分解为：
$$
\mathbb{E}_{(s_t, a_t) \sim \pi_{\theta'}} \left[ A^{\pi_\theta}(s_t, a_t) \right] = \sum_{s} P(s_t = s | \pi_{\theta'}) \sum_{a} \pi_{\theta'}(a|s) A^{\pi_\theta}(s, a)
$$
其中：

- $P(s_t = s | \pi_{\theta'})$ 是策略 $\pi_{\theta'}$ 下在时间 $t$ 访问状态 $s$ 的概率。
- $\pi_{\theta'}(a|s)$ 是策略 $\pi_{\theta'}$ 在状态 $s$ 选择动作 $a$ 的概率。

这个式子可以拆解成两个部分：

- **内层求和 $ \sum_{a} \pi_{\theta{\prime}}(a|s) A^{\pi_\theta}(s, a) $**
  - 表示给定状态 $s$，计算动作的加权优势，也就是对于每个状态 $s$，按照新的策略 $\pi_{\theta'}$ 选择动作 $a$，并计算优势函数 $A^{\pi_{\theta}}(s,a)$​。
  - $ A^{\pi_\theta}(s, a)$ 表示在状态 $s$ 下，动作 $a$ 相对于旧策略 $\pi_\theta$ 的优势。

- **外层求和：** $\sum_{s} P(s_t = s | \pi_{\theta{\prime}}) $，新策略下，状态 $s$​ 的访问频率



**3.代入交换求和顺序**
$$
=\sum_{t=0}^\infty \gamma^t \sum_{s} P(s_t = s | \pi_{\theta'}) \sum_{a} \pi_{\theta'}(a|s) A^{\pi_\theta}(s, a)
$$
交换求和顺序，先对 $s$ 求和，再对 $t$ 求和
$$
\sum_{s} \sum_{t=0}^\infty \gamma^t P(s_t = s | \pi_{\theta'}) \sum_{a} \pi_{\theta'}(a|s) A^{\pi_\theta}(s, a)
$$


**4.提取折扣状态访问分布**

折扣状态访问分布：表示策略 $\pi_{\theta'}$ 在折扣因子 $\gamma$ 下访问状态 $s$ 的长期概率（归一化后）。

根据折扣状态访问分布的定义：
$$
\nu^{\pi_{\theta'}}(s) = (1 - \gamma) \sum_{t=0}^\infty \gamma^t P(s_t = s | \pi_{\theta'})
$$
解得：
$$
\sum_{t=0}^\infty \gamma^t P(s_t = s | \pi_{\theta'}) = \frac{\nu^{\pi_{\theta'}}(s)}{1 - \gamma}
$$
代入 3 的表达式：
$$
\sum_{s} \sum_{t=0}^\infty \gamma^t P(s_t = s | \pi_{\theta'}) \sum_{a} \pi_{\theta'}(a|s) A^{\pi_\theta}(s, a)=\sum_{s} \frac{\nu^{\pi_{\theta'}}(s)}{1 - \gamma} \sum_{a} \pi_{\theta'}(a|s) A^{\pi_\theta}(s, a)
$$



**5.转为期望**

将求和符号转化为期望：
$$
\frac{1}{1 - \gamma} \sum_{s} \nu^{\pi_{\theta'}}(s) \sum_{a} \pi_{\theta'}(a|s) A^{\pi_\theta}(s, a) = \frac{1}{1 - \gamma} \mathbb{E}_{s \sim \nu^{\pi_{\theta'}}} \mathbb{E}_{a \sim \pi_{\theta'}} \left[ A^{\pi_\theta}(s, a) \right]
$$
- $\sum_{s} \nu^{\pi_{\theta'}}(s) \cdot \text{[...]}$ 对应 $\mathbb{E}_{s \sim \nu^{\pi_{\theta'}}}$。
- $\sum_{a} \pi_{\theta'}(a|s) \cdot \text{[...]}$ 对应 $\mathbb{E}_{a \sim \pi_{\theta'}}$。

这个式子的意思是：新策略 $\pi_{\theta'}$ 是否比旧策略 $\pi_{\theta}$ 更好，取决于在所有访问的 $s-a$ 下，优势函数的期望是否更大



想要直接优化这个公式很困难

**难点1：$\nu^{\pi_{\theta{\prime}}}(s)$ 依赖于 $\pi_{\theta'}$**

- 也就是说，要用新策略 $\pi_{\theta{\prime}}$ 来采样新的状态分布 $\nu^{\pi_{\theta{\prime}}}(s)$ ，**但问题是现在还没有这个新策略**
- 直觉上来理解，想评价一个新方法，但还没试过它，怎么能知道它的效果？
- 这就相当于：想换工作，但是只有在换了之后才能知道新的工作好不好，这种情况很麻烦。

**难点2：遍历所有可能的新策略不现实**

- 若要尝试所有可能的 $\pi_{\theta'}$ 然后逐个计算回报，这不可行，策略空间太大了



#### 用 $\pi_{\theta}$ 近似 $\pi_{\theta\prime}$

TRPO（信任区域策略优化） 采用了一种近似的方法来简化计算：

1. 假设新策略 $\pi_{\theta{\prime}}$ 和旧策略 $\pi_{\theta}$ 差别不大（即，每次更新策略的时候，不让它变化太大）
2. 如果策略变化不大，那么状态访问分布 $\nu^{\pi_{\theta{\prime}}}(s)$ 也不会变太多
   - 也就是说，新策略访问的状态分布和旧策略差不多。
   - 所以可以**用旧策略的状态分布** $\nu^{\pi_{\theta}}(s)$ **来近似** $\nu^{\pi_{\theta{\prime}}}(s) $。
   - 这样就不需要重新采样新的数据了，而是直接用旧的数据来估计新策略的效果

于是，优化目标变成：
$$
L_{\theta}(\theta{\prime}) = J(\theta) + \frac{1}{1 - \gamma} \mathbb{E}{s \sim \nu^{\pi{\theta}}} \mathbb{E}{a \sim \pi{\theta{\prime}}} [ A^{\pi_{\theta}}(s, a) ]
$$
这个和上面这个式子的区别在于：$v^{\pi_\theta^{\prime}}$ 换成了 $v^{\pi_{\theta}}$

这个公式的意思是：

**用旧策略 $\pi_{\theta}$ 的状态访问分布 $\nu^{\pi_{\theta}}(s)$ 来计算新策略的期望，避免了直接计算 $\nu^{\pi_{\theta{\prime}}}(s) $的麻烦。**



#### 重要性采样修正

状态分布用 $\pi_{\theta}$ 来近似，但是动作是由新策略 $\pi_{\theta'}$ 采样的，若直接用旧策略 $\pi_{\theta}$​ 采样的动作来估计，新策略的效果，可能会有偏差。



采用重要性采样来修正偏差，即：
$$
\mathbb{E}{\pi{\theta{\prime}}} [ A^{\pi_\theta}(s, a) ] = \mathbb{E}{\pi\theta} \left[ \frac{\pi_{\theta{\prime}}(a | s)}{\pi_\theta(a | s)} A^{\pi_\theta}(s, a) \right]
$$

$$
L_{\theta}(\theta{\prime}) = J(\theta) + \mathbb{E}{s \sim \nu^{\pi{\theta}}} \mathbb{E}{a \sim \pi{\theta}} \left[ \frac{\pi_{\theta{\prime}}(a | s)}{\pi_{\theta}(a | s)} A^{\pi_{\theta}}(s, a) \right]
$$



此外，还需要保证新旧策略足够接近，TRPO 用 **KL 散度**来衡量策略之间的距离，确保新旧策略状态访问分布变化很小。



#### 优化目标

TRPO 提出的优化目标是：
$$
\max_{\theta{\prime}} L_{\theta}(\theta{\prime}) \quad \text{subject to} \quad \mathbb{E}{s \sim \nu^{\pi{\theta}}} \left[ D_{\text{KL}} (\pi_{\theta_k}(\cdot | s), \pi_{\theta{\prime}}(\cdot | s)) \right] \leq \delta
$$
**目标函数（优化目标）**：$\max_{\theta{\prime}} L_{\theta}(\theta{\prime})$

这里 $L_{\theta}(\theta{\prime})$ 是一个替代目标函数，用于衡量新策略$ \pi_{\theta{\prime}} $相对于旧策略$\pi_{\theta} $的改进程度。



**约束条件（限制策略更新幅度）**：$\mathbb{E}{s \sim \nu^{\pi{\theta}}} \left[ D_{\text{KL}} (\pi_{\theta_k}(\cdot | s), \pi_{\theta{\prime}}(\cdot | s)) \right] \leq \delta$

这里 $D_{\text{KL}}$ 用于衡量新策略$ \pi_{\theta{\prime}}$ 与旧策略 $\pi_{\theta}$ 在相同状态下的不同程度

- 期望的状态分布 $s \sim \nu^{\pi_{\theta}}(s)$ 表示根据旧策略 $\pi_{\theta}$ 访问各个状态的频率

- 约束条件要求：新策略 $\pi_{\theta{\prime}}$不能离旧策略 $ \pi_{\theta}$ 太远，即 **两者的 KL 散度不能超过某个阈值** $\delta$

这个优化问题的核心思想是：

- 希望找到一个更优的新策略 $\pi_{\theta{\prime}}$ ，使得它比旧策略$ \pi_{\theta} $具有更高的收益（即优化 $L_{\theta}(\theta{\prime}$) ）。
- 但同时，不希望新策略与旧策略相差太大，避免剧烈的策略变化带来的不稳定性（即约束 KL 散度不超过$ \delta $）



不等式约束定义了策略空间中的一个 KL 球，被称为信任区域。

在这个区域中，可以认为当前学习的策略和环境交互的状态分布于上一轮策略最后采样的状态分布一致。

可以基于一步行动的重要性采样方法使当前学习策略稳定提升。



#### 近似计算 KL 散度

> **为什么 TRPO 中要近似 KL 散度**
>
> 在深度神经网络中的参数数目非常的大，计算 KL 散度的计算代价很高。
>
> 1. 若是离散动作空间，$\pi_{\theta}(a|s) = \frac{\exp(f_{\theta}(s, a))}{\sum_{a{\prime}} \exp(f_{\theta}(s, a{\prime}))}$
>
> 对于每个状态 $s$，离散部分的 KL 散度可以直接计算：
> $$
> D_{\text{KL}}(\pi_{\theta} || \pi_{\theta{\prime}}) = \sum_{a} \pi_{\theta}(a | s) \log \frac{\pi_{\theta}(a | s)}{\pi_{\theta{\prime}}(a | s)}
> $$
> 在神经网络中，给定一批样本 $s_i$，用蒙特卡洛近似：
> $$
> \hat{D}{\text{KL}} = \frac{1}{N} \sum{i=1}^{N} \sum_{a} \pi_{\theta}(a | s_i) \log \frac{\pi_{\theta}(a | s_i)}{\pi_{\theta{\prime}}(a | s_i)}
> $$
>
> 2. 若是连续动作空间，通常假设策略服从高斯分布：$\pi_{\theta}(a | s) = \mathcal{N}(\mu_{\theta}(s), \Sigma_{\theta}(s))$
>
> -  $\mu_{\theta}(s)$ 是神经网络输出的均值
> - $\Sigma_{\theta}(s)$ 是对角协方差矩阵（通常表示为对数标准差 $\log \sigma_{\theta}$ ）。
>
> 假设新策略 $\pi_{\theta{\prime}}$ 也是高斯分布 $\mathcal{N}(\mu_{\theta{\prime}}, \Sigma_{\theta{\prime}})$ ，那么两者的 KL 散度有解析解：
> $$
> D_{\text{KL}} (\mathcal{N}(\mu_{\theta}, \Sigma_{\theta}) || \mathcal{N}(\mu_{\theta{\prime}}, \Sigma_{\theta{\prime}})) = \frac{1}{2} \sum_i \left[ \frac{\sigma_{\theta{\prime}, i}^2}{\sigma_{\theta, i}^2} + \frac{(\mu_{\theta, i} - \mu_{\theta{\prime}, i})^2}{\sigma_{\theta, i}^2} - 1 + \log \frac{\sigma_{\theta, i}^2}{\sigma_{\theta{\prime}, i}^2} \right]
> $$
> 

在 TRPO 中，为了避免计算 KL 散度的计算代价，采用了二阶近似来处理：
$$
D_{\text{KL}}(\pi_{\theta} || \pi_{\theta + \Delta \theta}) \approx \frac{1}{2} \Delta\theta^T H \Delta\theta
$$


对于多元函数 $f(\theta)$ ，在 $\theta_k$ 处的泰勒展开式为:
$$
f(\theta) \approx f(\theta_k) + g^T (\theta - \theta_k) + \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)
$$
其中：

- $g$ 是梯度： $g = \nabla_{\theta} f(\theta)$ 表示函数在 $\theta_k$处的变化速率（方向和大小）。
- H 是黑塞矩阵： 二阶导数 的矩阵，表示函数在 $\theta_k$ 处的曲率，即函数的变化是如何加速或减速的。

$$
H = \begin{bmatrix}
\frac{\partial^2 f}{\partial \theta_1^2} & \frac{\partial^2 f}{\partial \theta_1 \partial \theta_2} \\
\frac{\partial^2 f}{\partial \theta_2 \partial \theta_1} & \frac{\partial^2 f}{\partial \theta_2^2}
\end{bmatrix}
$$

由于 KL 散度是关于参数 $\theta$ 的光滑函数，可以使用泰勒展开进行近似：
$$
D_{\text{KL}} (\pi_{\theta} || \pi_{\theta + \Delta\theta}) \approx D_{\text{KL}} (\pi_{\theta} || \pi_{\theta}) + \nabla_{\theta} D_{\text{KL}} (\pi_{\theta})^T \Delta\theta + \frac{1}{2} \Delta\theta^T H \Delta\theta
$$
其中：

- **第一项** $D_{\text{KL}} (\pi_{\theta} || \pi_{\theta})$ = 0 （因为 KL 散度计算同一个分布的结果为 0）
- **第二项** $\nabla_{\theta} D_{\text{KL}} (\pi_{\theta})^T \Delta\theta$ 是 KL 散度对参数的梯度，但是由于 $\theta$ 和 $\theta{\prime}$ 很接近，KL 散度在 $\theta$ 处的梯度通常接近0，因此这一项可以忽略
- **第三项** $\frac{1}{2} \Delta\theta^T H \Delta\theta$ 取决于Hessian 矩阵（即 KL 散度对参数的二阶导数），是二阶近似项。



- $g$，告诉如何沿着最快的方向更新参数
- $H$，告诉沿着梯度的方向更新的幅度
  - 如果 Hessian 矩阵的对角元素很大，说明KL 散度的变化非常敏感，我们就应该减小步长
  - 如果 Hessian 矩阵的对角元素很小，说明KL 散度的变化不太敏感，我们就应该加大步长



通过二阶近似，KL 散度的约束可以转化为一个二次约束，形式为：
$$
\frac{1}{2} \Delta\theta^T H \Delta\theta \leq \delta
$$


#### KKT条件

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
$$
\text{KL}(\theta_k \| \theta') \approx \frac{1}{2} (\theta' - \theta_k)^T H (\theta' - \theta_k) \leq \delta
$$
将约束优化问题转化为无约束形式，引入拉格朗日乘子 $\lambda $：
$$
\mathcal{L}(\theta', \lambda) = L_{\theta_k}(\theta') - \lambda \left( \frac{1}{2} (\theta' - \theta_k)^T H (\theta' - \theta_k) - \delta \right)
$$
其中：
- $L_{\theta_k}(\theta')$  是替代优势函数（目标函数的一阶近似）。
- 约束项为  $\frac{1}{2} (\theta' - \theta_k)^T H (\theta' - \theta_k) - \delta \leq 0 $



根据KKT条件，需满足以下条件：
1. **梯度条件**：对 $\theta'$  求导并设为零：
   $$
   \nabla_{\theta'} L_{\theta_k}(\theta') - \lambda H (\theta' - \theta_k) = 0
   $$
   假设在 $\theta_k$ 附近，目标函数梯度为 $g = \nabla_{\theta} L_{\theta_k}(\theta) \big|_{\theta=\theta_k}$ ，则方程变为：
   $$
   g - \lambda H (\theta' - \theta_k) = 0
   $$
   解得参数更新方向：
   $$
   \theta' = \theta_k + \lambda^{-1} H^{-1} g
   $$

2. **约束条件**：代入KL散度约束：
   $$
   \frac{1}{2} (\theta' - \theta_k)^T H (\theta' - \theta_k) = \delta
   $$
   将  $\theta' - \theta_k = \lambda^{-1} H^{-1} g$  代入约束，解得：
   $$
   \lambda = \sqrt{\frac{g^T H^{-1} g}{2\delta}}
   $$

3. **互补松弛条件**：$\lambda \geq 0$ ，且当约束严格不等式成立时 $\lambda = 0$ 。但在TRPO中，约束通常处于边界 $\lambda > 0$ 



结合梯度条件和约束条件，最终更新公式为：
$$
\theta' = \theta_k + \underbrace{\sqrt{\frac{2\delta}{g^T H^{-1} g}}}_{\text{步长}} H^{-1} g
$$
其中：
- $H^{-1} g$ 是自然梯度方向，表明调整后的更新方向，根据曲率调整更新方向，保证沿着梯度方向更新，而且步长也被适当的调整，避免 KL 散度过大。
- 步长由 $\delta$  和梯度的模长 $g^T H^{-1} g$  共同决定。



#### 共轭梯度

计算和存储黑塞矩阵的逆十分浪费资源，TRPO 通过共轭梯度法回避了这个问题。



设 $x = H^{-1}g$，这个 $x$ 就是参数更新方向。

由 KL 散度约束有：
$$
\frac{1}{2} (\theta' - \theta_k)^T H (\theta' - \theta_k) = \delta
$$
设最大的步长为 $\beta$，则
$$
\frac{1}{2}(\beta x)^TH(\beta x)=\delta
$$
解得：
$$
\beta ={\sqrt{\frac{2\delta}{x^T H x}}}
$$

$$
\theta_{k+1}=\theta+{\sqrt{\frac{2\delta}{x^T H^{-1} x}}}x
$$
所以只要计算 $x=H^{-1}g$ 就可以实现更新参数了，问题就转化为求解线性方程组 $Hx=g$



**具体流程**

1. **初始化**：设定初始点 x_0 和初始残差 r_0 。在 TRPO 中，初始点 x_0 通常是自然梯度的初始估计（一般可以设置为零向量），而残差 r_0 是根据 b 和初始点 x_0 计算的。
   - $r_0$ = $g - H x_0$ 
   - $x_0$ 是初始的自然梯度估计（可以设置为零向量）。
   - 残差 $r_0$ 反映了当前解与目标的差距，表示当前的误差。

2. **设置初始搜索方向**：
   - $p_0 = r_0$
   - 初始搜索方向 $p_0$ 设为初始残差 $r_0$ ，因为一开始搜索方向就应该沿着残差方向进行，尝试修正当前解。

3. **迭代更新**：在每次迭代中，计算 **步长**，更新解，更新残差，并检查停止条件。
   - 计算步长 $\alpha_k$ ：它控制了每一步更新的大小。步长可以通过下面的公式计算：
   - $\alpha_k = \frac{r_k^T r_k}{p_k^T H p_k}$
   - 其中：
     - $r_k$ 是第 k 次迭代的 残差，它表示当前解与目标的差距。
     - $p_k$ 是第 k 次迭代的搜索方向，表示我们沿着哪个方向进行更新。
     - $H $是Hessian 矩阵，目标函数的二阶导数矩阵，它反映了目标函数在各个方向上的曲率。
   - 这个公式计算残差和搜索方向与Hessian 矩阵 的乘积，调整每一步的步长，确保沿着正确的方向更新，同时避免步长过大或过小。
   - **更新解**：
     - $x_{k+1} = x_k + \alpha_k p_k$
     - 这个公式表示我们根据步长 $\alpha_k$ 沿着搜索方向 $p_k$ 进行更新。这样，新的解 $x_{k+1}$ 就是当前解 $x_k$ 加上沿着 自然梯度方向更新的量。
   - **更新残差**：
     - $r_{k+1} = r_k - \alpha_k H p_k$
     - 更新后的残差 $r_{k+1}$ 表示当前解 $x_{k+1}$ 与目标之间的差距。它通过减去 $\alpha_k H p_k$ 来计算。
   - **检查停止条件**：
     - 如果残差 $r_{k+1}$ 足够小，意味着当前解已经足够接近目标解，则停止迭代。通常，停止条件是：
     - $\| r_{k+1} \| \text{ is small enough}$
     - 这表示如果残差小于某个阈值，我们就可以停止迭代，认为已经找到了一个近似解
   - **更新搜索方向**：
     - $p_{k+1} = r_{k+1} + \beta_k p_k$
     - 这里 $\beta_k$​ 是一个调整因子，通常通过 **Fletcher-Reeves 或 Polak-Ribière** 等公式计算，它帮助我们根据当前的残差调整新的搜索方向。
     - 这是为了使得新的搜索方向不仅考虑当前的残差 $r_{k+1}$ ，还考虑之前的搜索方向
4. **停止迭代**：当残差 $r_k$ 小于某个阈值时，表示解已经足够精确，达到停止条件，可以退出迭代。



Hessian 矩阵可能是一个百万维度×百万维度的矩阵，计算 Hessian 矩阵并存储它的所有元素代价很高。

假设有一个目标函数 $f(\theta)$，Hessian 矩阵 $H$ 定义为：
$$
H = \nabla^2_{\theta} f(\theta)
$$
计算Hessian 矩阵与向量 $v$ 的乘积 $H v$ ，要使用以下表达式：
$$
Hv = \nabla_{\theta} \left( \nabla_{\theta} f(\theta)^T v \right)
$$

- 即先计算 KL 散度的梯度 $ \nabla_{\theta} D_{\text{KL}} $
- 在计算这个梯度与向量 $v$ 的点乘
- 最后再对这个点乘的结果计算梯度，得到 $Hv$



#### 线性搜索

TRPO 通过二阶泰勒展开近似 KL 散度约束，并使用共轭梯度法计算自然梯度更新方向。但是，二阶近似不是严格的全局求解，只是局部的近似解。

因此，直接使用这个解进行策略更新可能会出现两个问题：

- **更新后的策略** $\theta{\prime}$ **可能比当前策略** $\theta_k$ **更差** —— 也就是说，目标函数 $L_{\theta_k} $可能没有真正提升。_
- **可能违反 KL 散度约束** —— 由于 Hessian 近似误差，可能导致 KL 散度增量 $D_{\text{KL}} (\pi_{\theta_k}, \pi_{\theta{\prime}})$ 超过预设的阈值 $\delta$ 

为了避免这些问题，TRPO 在最终更新参数前，加入了一步线性搜索，确保新的策略 $\theta_{k+1}$ 既能提升目标函数 $L_{\theta_k} $，又满足 KL 散度约束。



线性搜索不会直接使用计算得到的自然梯度更新步长，而是通过指数衰减进行逐步尝试，寻找一个合适的步长：
$$
\theta_{k+1} = \theta_k + \alpha^i \sqrt{\frac{2\delta}{x^T H x}} x
$$
其中：

- $\alpha \in (0,1)$ 是一个超参数，通常设为 0.5（意味着每次步长缩小 50%）。
- $i$ 是最小的非负整数，表示我们在不断缩小步长，直到满足条件。
- $x$ **是共轭梯度法求出的自然梯度方向**，即一个近似的更新方向。
- $\sqrt{\frac{2\delta}{x^T H x}}$ **确保在未调整** $\alpha$ **时，步长能恰好满足 KL 散度约束。**



换句话说，会先尝试一个较大的步长，然后如果更新后的策略不符合 KL 约束或目标函数下降了，就不断缩小步长，直到找到一个合适的 $i$ ，使得：

- $D_{\text{KL}} (\pi_{\theta_k}, \pi_{\theta_{k+1}}) \leq \delta$ （满足 KL 约束）
- $L_{\theta_{k+1}} > L_{\theta_k}$ （目标函数变大）



#### 广义优势估计 GAE

$$
A^{\pi}(s, a) = Q^{\pi}(s, a) - V^{\pi}(s)
$$

单步 TD 误差定义为：
$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$
GAE 的核心思想是 结合多步 TD 误差，通过指数加权平均不同步数的估计：
$$
\begin{align*}
A_t^{(1)} &= \delta_t \\
&= -V(s_t) + r_t + \gamma V(s_{t+1}) \\
A_t^{(2)} &= \delta_t + \gamma \delta_{t+1} \\
&= -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) \\
A_t^{(3)} &= \delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2} \\
&= -V(s_t) + r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \gamma^3 V(s_{t+3}) \\
&\vdots \\
A_t^{(k)} &= \sum_{l=0}^{k-1} \gamma^l \delta_{t+l} \\
&= -V(s_t) + r_t + \gamma r_{t+1} + \cdots + \gamma^{k-1} r_{t+k-1} + \gamma^k V(s_{t+k})
\end{align*}
$$
然后，GAE 将这些不同步数的优势估计进行指数加权平均：
$$
\begin{align*}
A_t^{GAE} &= (1 - \lambda)(A_t^{(1)} + \lambda A_t^{(2)} + \lambda^2 A_t^{(3)} + \cdots) \\
&= (1 - \lambda)(\delta_t + \lambda (\delta_t + \gamma \delta_{t+1}) + \lambda^2 (\delta_t + \gamma \delta_{t+1} + \gamma^2 \delta_{t+2}) + \cdots) \\
&= (1 - \lambda)(\delta (1 + \lambda + \lambda^2 + \cdots) + \gamma \delta_{t+1} (\lambda + \lambda^2 + \lambda^3 + \cdots) + \gamma^2 \delta_{t+2} (\lambda^2 + \lambda^3 + \lambda^4 + \cdots) + \cdots) \\
&= (1 - \lambda) \left( \delta_t \frac{1}{1 - \lambda} + \gamma \delta_{t+1} \frac{\lambda}{1 - \lambda} + \gamma^2 \delta_{t+2} \frac{\lambda^2}{1 - \lambda} + \cdots \right) \\
&= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
\end{align*}
$$

- 当 $\lambda=1$ 时，GAE 退化成 MC 估计，高方差，低偏差

- 当 $\lambda=0$ 时，GAE 退化成 TD 估计，低方差，高偏差

为了方便计算，GAE 可以用递归形式表示：
$$
A_t^{\text{GAE}(\gamma, \lambda)} = \delta_t + (\gamma \lambda) \cdot A_{t+1}^{\text{GAE}(\gamma, \lambda)}
$$
