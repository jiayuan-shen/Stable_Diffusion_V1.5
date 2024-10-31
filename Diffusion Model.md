# Diffusion Model

Diffusion models are inspired by non-equilibrium thermodynamics. They define a Markov chain of diffusion steps to slowly add random noise to data and then learn to reverse the diffusion process to construct desired data samples from the noise. Unlike VAE or flow models, diffusion models are learned with a fixed procedure and the latent variable has high dimensionality (same as the original data).

<img src="../../Library/Application Support/typora-user-images/截屏2024-07-27 上午1.00.58.png" alt="截屏2024-07-27 上午1.00.58" style="zoom:50%;" />

# What are Diffusion Models?

Several diffusion-based generative models have been proposed with similar ideas underneath, including *diffusion probabilistic models* ([Sohl-Dickstein et al., 2015](https://arxiv.org/abs/1503.03585)), *noise-conditioned score network* (**NCSN**; [Yang & Ermon, 2019](https://arxiv.org/abs/1907.05600)), and *denoising diffusion probabilistic models* (**DDPM**; [Ho et al. 2020](https://arxiv.org/abs/2006.11239)).

The easiest way to think of a Variational Diffusion Model (VDM) [4, 5, 6] is simply as a Markovian Hierarchical Variational Autoencoder with three key restrictions: 

- The latent dimension is exactly equal to the data dimension. 
- The structure of the latent encoder at each timestep is not learned; it is pre-defined as a linear Gaussian model. In other words, it is a Gaussian distribution centered around the output of the previous timestep. 
- The Gaussian parameters of the latent encoders vary over time in such a way that the distribution of the latent at final timestep $T$ is a standard Gaussian

# Some Piror Knowledge

## Jensen's inequality

In the context of [probability theory](https://en.wikipedia.org/wiki/Probability_theory), it is generally stated in the following form: if $X$ is a [random variable](https://en.wikipedia.org/wiki/Random_variable) and $\phi$ is a convex function, then
$$
\phi(\mathbb{E}[X]) \le \mathbb{E}[\phi(X)]
$$
if $\phi$ is a concave function, then:
$$
\phi(\mathbb{E}[X]) \ge \mathbb{E}[\phi(X)]
$$
This inequality can be used to derive the VAE ELOB but not necessarily. 



## Hierarchical Variational Autoencoders

A Hierarchical Variational Autoencoder (HVAE) [2, 3] is a generalization of a VAE that extends to multiple hierarchies over latent variables. Under this formulation, latent variables themselves are interpreted as generated from other higher-level, more abstract latents. Intuitively, just as we treat our three-dimensional observed objects as generated from a higher-level abstract latent, the people in Plato’s cave treat three dimensional objects as latents that generate their two-dimensional observations. Therefore, from the perspective of Plato’s cave dwellers, their observations can be treated as modeled by a latent hierarchy of depth two (or more).

Whereas in the general HVAE with $T$ hierarchical levels, each latent is allowed to condition on all previous latents, in this work we focus on a special case which we call a Markovian HVAE (MHVAE). In a MHVAE, the generative process is a Markov chain; that is, each transition down the hierarchy is Markovian, where

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-28 上午1.06.51.png" alt="截屏2024-09-28 上午1.06.51" style="zoom:50%;" />

Decoding each latent $z_t$ only conditions on previous latent $z_{t+1}$. Intuitively, and visually, this can be seen as simply stacking VAEs on top of each other, as depicted in Figure 2; another appropriate term describing this model is a Recursive VAE. Mathematically, we represent the joint distribution and the posterior of a Markovian HVAE as:

$$
p(x, z_{1:T}) = p(z_T)p(x|z_1) \prod_{t=2}^T p(z_{t-1}|z_t) \tag{23}
$$

$$
q_\phi(z_{1:T}|x) = q_\phi(z_1|x) \prod_{t=2}^T q_\phi(z_t|z_{t-1}) \tag{24}
$$

Then, we can easily extend the ELBO to be:

$$
\begin{align}
\log p(x) &= \log \int p(x, z_{1:T}) dz_{1:T} \tag{25}\\
&= \log \int \frac{p(x, z_{1:T}) q_\phi(z_{1:T}|x)}{q_\phi(z_{1:T}|x)} dz_{1:T} \tag{26} \quad \text{(Multiply by } 1 = \frac{q_\phi(z_{1:T}|x)}{q_\phi(z_{1:T}|x)}\text{)}\\
&= \log \mathbb{E}_{q_\phi(z_{1:T}|x)} \left[ \frac{p(x, z_{1:T})}{q_\phi(z_{1:T}|x)} \right] \tag{27} \quad \text{(Definition of Expectation)}\\
&\geq \mathbb{E}_{q_\phi(z_{1:T}|x)} \left[ \log \frac{p(x, z_{1:T})}{q_\phi(z_{1:T}|x)} \right] \tag{28} \quad \text{(Apply Jensen’s Inequality)}

\end{align}
$$



We can then plug our joint distribution (Equation 23) and posterior (Equation 24) into Equation 28 to produce an alternate form:

$$
\mathbb{E}_{q_\phi(z_{1:T}|x)} \left[ \log \frac{p(x, z_{1:T})}{q_\phi(z_{1:T}|x)} \right] = \mathbb{E}_{q_\phi(z_{1:T}|x)} \left[ \log \frac{p(z_T)p(x|z_1) \prod_{t=2}^T p(z_{t-1}|z_t)}{q_\phi(z_1|x) \prod_{t=2}^T q_\phi(z_t|z_{t-1})} \right] \tag{29}
$$

As we will show below, when we investigate Variational Diffusion Models, this objective can be further decomposed into interpretable components.

## PDF of the Standard Gaussian Distribution

$$
\begin{align}
f(x|\mu,\sigma^2) &= \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)\\
&= \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2}\right)
\end{align}
$$



## Forward diffusion process

Given a data point sampled from a real data distribution $\mathbf{x}_0 \sim q(\mathbf{x})$ , let us define a *forward diffusion process* in which we add small amount of Gaussian noise to the sample in $T$ steps, producing a sequence of noisy samples $\mathbf{x}_1, \dots, \mathbf{x}_T$ . The step sizes are controlled by a variance schedule $\{\beta_t \in (0, 1)\}_{t=1}^T$ . The variance parameter $\beta_t$ can be fixed to a constant or chosen as a schedule over the $T$ timesteps. In fact, one can define a variance schedule, which can be linear, quadratic, cosine etc. The original DDPM authors utilized a linear schedule increasing from $\beta_1=10^{-4}$ to $\beta_2=0.02$. [Nichol et al. 2021](https://arxiv.org/abs/2102.09672) showed that employing a cosine schedule works even better.

<img src="../../Library/Application Support/typora-user-images/截屏2024-10-01 下午6.25.23.png" alt="截屏2024-10-01 下午6.25.23" style="zoom:50%;" />

Given a data-point $\mathbf{x}_0$ sampled from the real data distribution $q(\mathbf{x})$, where $\mathbf{x}_0 \sim q(\mathbf{x})$, one can define a forward diffusion process by adding noise. Specifically, at each step of the Markov chain we add Gaussian noise with variance $\beta_t$ to $\mathbf{x}_{t-1}$, producing a new latent variable $\mathbf{x}_t$ with distribution $q(\mathbf{x}_t|\mathbf{x}_{t-1})$. This diffusion process can be formulated as follows:
$$
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
The data sample $\mathbf{x}_0$ gradually loses its distinguishable features as the step $t$ becomes larger. Eventually when $T \to \infty$, $\mathbf{x}_T$ is equivalent to an isotropic Gaussian distribution.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-28 下午10.40.26.png" alt="截屏2024-09-28 下午10.40.26" style="zoom:50%;" />

Figure 3: A visual representation of a Variational Diffusion Model; $x_0$ represents true data observations such as natural images, $x_T$ represents pure Gaussian noise, and $x_t$ is an intermediate noisy version of $x_0$. Each $q(x_t \mid x_{t-1})$ is modeled as a Gaussian distribution that uses the output of the previous state as its mean.

Since we are in the multi-dimensional scenario $\mathbf{I}$ is the identity matrix, indicating that each dimension has the same variance $\beta_t$. Note that $q(\mathbf{x}_t|\mathbf{x}_{t-1})$ is still a normal distribution, defined by the mean $\boldsymbol{\mu}$ and the variance $\boldsymbol{\Sigma}$ where $\boldsymbol{\mu}_t = \sqrt{1-\beta_t}\mathbf{x}_{t-1}$ and $\boldsymbol{\Sigma}_t=\beta_t\mathbf{I}$will always be a diagonal matrix of variances. 

Thus, we can go in a closed form from the input data $\mathbf{x}_0$ to $\mathbf{x}_T$ in a tractable way. Mathematically, this is the posterior probability and is defined as:
$$
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
$$
The symbol: $q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) =q(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \cdots, \mathbf{x}_T \vert \mathbf{x}_0) =q(\mathbf{x}_1 \vert \mathbf{x}_0) \times q(\mathbf{x}_2 \vert \mathbf{x}_1) \times \dots \times q(\mathbf{x}_T \vert \mathbf{x}_{T-1})$. It's also called trajectory.

> ## How we could expand the Joint Distribution Without the Markov Assumption
>
> ### **For $ T = 5 $, the expansion is:**
>
> Given $ \mathbf{x}_0 $, the joint distribution $ q(\mathbf{x}_{1:5} \vert \mathbf{x}_0) $ expands as:
>
> $$
> \begin{align*}
> q(\mathbf{x}_{1:5} \vert \mathbf{x}_0) &= q(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \mathbf{x}_4, \mathbf{x}_5 \vert \mathbf{x}_0) \\
> &= q(\mathbf{x}_1 \vert \mathbf{x}_0) \times q(\mathbf{x}_2 \vert \mathbf{x}_0, \mathbf{x}_1) \times q(\mathbf{x}_3 \vert \mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2) \times q(\mathbf{x}_4 \vert \mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3) \times q(\mathbf{x}_5 \vert \mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \mathbf{x}_4)
> \end{align*}
> $$
>
> **Explanation:**
>
> - **$ q(\mathbf{x}_1 \vert \mathbf{x}_0) $:** The probability of transitioning from $ \mathbf{x}_0 $ to $ \mathbf{x}_1 $.
> - **$ q(\mathbf{x}_2 \vert \mathbf{x}_0, \mathbf{x}_1) $:** The probability of transitioning to $ \mathbf{x}_2 $ given both $ \mathbf{x}_0 $ and $ \mathbf{x}_1 $.
> - **And so on for subsequent terms.**
>
> ### **General Form (Without Markov Assumption):**
>
> For any $ T $:
>
> $$
> q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = q(\mathbf{x}_1 \vert \mathbf{x}_0) \prod_{t=2}^T q(\mathbf{x}_t \vert \mathbf{x}_0, \mathbf{x}_1, \dots, \mathbf{x}_{t-1})
> $$
>
> ## Key Points to Remember
>
> ### **a. Conditional Probabilities Involving Multiple Variables**
>
> - The notation $ p(A \vert B, C, D) $ means that the probability of $ A $ is evaluated considering that $ B $, $ C $, and $ D $ are known.
>
> ### **b. Chain Rule with Conditional Probabilities**
>
> - When expanding a joint conditional probability, we apply the chain rule by successively conditioning on more variables:
>
>   $$
>   p(A, B, C \vert D) = p(A \vert D) \times p(B \vert A, D) \times p(C \vert A, B, D)
>   $$
>
> ### **c. Importance of the Order**
>
> - The order in which variables are conditioned matters. In our expansion, each $ \mathbf{x}_t $ is conditioned on all previous $ \mathbf{x}_i $ (for $ i < t $).
>
> ### **d. Non-Markovian Processes**
>
> - Without the Markov assumption, we cannot simplify the dependencies, and each conditional probability must consider the entire history.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-29 上午12.24.40.png" alt="截屏2024-09-29 上午12.24.40" style="zoom:50%;" />

### **Variance Schedule $ \{ \beta_t \}_{t=1}^T $**:

- $ \beta_t $ is a small positive number controlling the amount of noise added at each step $ t $.
- The sequence $ \{ \beta_t \} $ is often called the **noise schedule** or **variance schedule**.
- Typically, $ \beta_t $ increases from a small value to a larger value over $ t $.
- $\beta_t$ increased from 0.0001 to 0.02

### Parameterize the Gaussian encoder

We define a new parameter $\alpha_t = 1-\beta_t$, as $\beta_t$ generally getting smaller. 

We define the process of how to add the noise as we mentioned  in the piror knowledge:
$$
\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} \tag{1}\\
\text{OR}\\
\mathbf{x}_t = \sqrt{1 - \beta_t} \mathbf{x}_{t-1} + \sqrt{\beta_t} \boldsymbol{\epsilon}_{t-1}
$$

- $ \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) $
- **Interpretation**:
  - Scale down $ \mathbf{x}_{t-1} $ by $ \sqrt{1 - \beta_t} $.
  - Add Gaussian noise with standard deviation $ \sqrt{\beta_t} $.
  - As $t$ getting larger, $\alpha_t$ getting smaller, so does $\sqrt{\alpha_t}$ , but $\sqrt{1-\alpha_t}$ getting larger. Therefore, as we keep progress the forward diffussion, noise dominate the weight(More noise). 
  - All noise $\boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$

Therefore we could easily find out the correct equation of the $\mathbf{x}_{t-1}$
$$
\mathbf{x}_{t-1} = \sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-2}\tag{2}
$$

Given these two formulas we could easily calculate $\mathbf{x}_t$ from $\mathbf{x}_0$:

We could did this process recursively but it is generally slow, in the forward diffusion, we only care about the last stage $\mathbf{x}_T$ a image that with isotropic noise. 

##### Recursive method:

$$
\begin{align}
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) &= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} \\
q(\mathbf{x}_{t-1} \vert \mathbf{x}_{t-2}) &= \sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-2}\\
&\dots\dots\\
&\text{Until $\mathbf{x}_0$}

\end{align}
$$

##### But this method is too slow, we could calculate $\mathbf{x}_t$ directly from $\mathbf{x}_0$

$$
\begin{align}
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) &= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} \tag{1}\\
q(\mathbf{x}_{t-1} \vert \mathbf{x}_{t-2}) &= \sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}\tag{2}\\

\mathbf{x}_t &= \sqrt{\alpha_t} \big(\sqrt{\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{1-\alpha_{t-1}}\boldsymbol{\epsilon}_{t-2}\big) + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}
& \text{Subsitute (2) into (1)}\\
\mathbf{x}_t &= \sqrt{\alpha_t\alpha_{t-1}}\mathbf{x}_{t-2}+\sqrt{\alpha_t(1-\alpha_{t-1})}\boldsymbol{\epsilon}_{t-2} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1}
& \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \text{(.)} \\
\mathbf{x}_t&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
\end{align}
$$

Therefore, in terms of this formula, we can calculate any $\mathbf{x}_t$ directly from $\mathbf{x}_0$:

Let us define a new parameter for simplicity: $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$

$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}$ and $q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$

> ## Some Essential Interpretation(.)
>
> Recall (.) that suppose we have a random variable $\mathbf{x}$, and we do a random sample of this random variable $\mathbf{x}$, and have $x_1, x_2, \dots, x_n$, which they are i.i.d and $\sim \mathcal{N}(0,1)$. Now Suppose we add a scaler constant and multiply a scaler constant for all these random samples, How the orginal distribution of the random variable $\mathbf{x}$ will change to another distribution $\mathbf{Y}$?
>
> $\mathbf{Y}  = C\mathbf{x} +b$
>
> ##### How the distribution will change?
>
> - $\mu$ Mean
>   - $\mathbb{E}[\mathbf{Y}]=\mathbb{E}[c\mathbf{x}+b]=c\mathbb{E}[\mathbf{x}]+b=c \cdot 0+b=b$
> - $\sigma^2$ Variance:
>   - $\text{Var}(\mathbf{Y}) = \text{Var}(c\mathbf{x}+b)=c^2\text{Var}(\mathbf{x})=c^2\cdot1=c^2$
>
> Therefore, if we simulate the previous process into our (.) step we could get(Note: Even the $\boldsymbol{\epsilon}_\cdots$ are multivariate standard Gaussian Distribution, the previous derivative procedure still hold):
>
> $\boldsymbol{\epsilon}_{t-1}$ change from $\boldsymbol{\epsilon}_{t-1} \sim\mathcal{N}(\mathbf{0}, \mathbf{I})$ to $\boldsymbol{\epsilon}_{t-1} \sim\mathcal{N}(\mathbf{0}, \mathbf{1-\boldsymbol\alpha_t})$
>
> $\boldsymbol{\epsilon}_{t-2}$ change from $\boldsymbol{\epsilon}_{t-1} \sim\mathcal{N}(\mathbf{0}, \mathbf{I})$ to $\boldsymbol{\epsilon}_{t-1} \sim\mathcal{N}(\mathbf{0}, \boldsymbol {\mathbf{\alpha_t(1-\alpha_{t-1}})})$
>
> ## Some Essential Interpretation(*)
>
> (*) Recall that when we merge two Gaussians with different variance, $\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I})$ and $\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I})$, the new distribution is $\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I})$. Here the merged **standard deviation** is $\sqrt{(1 - \alpha_t) + \alpha_t (1-\alpha_{t-1})} = \sqrt{1 - \alpha_t\alpha_{t-1}}$.




> # Some Additional Details about the Forward Diffusion Process
>
> $q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$
>
> $q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha}_t \mathbf{x}_{t-1}, (1 - \alpha_t)\mathbf{I})$
>
> $q(\mathbf{x}_t \vert \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$





## Reverse diffusion process

### **The Challenge**

- **Objective**: Given that we can transform data into noise, how can we go from noise back to data?
- **Problem**: The reverse of the forward diffusion process is not directly accessible because it involves inverting the stochastic process.

As $T \rightarrow \infin$, the latent $\mathbf{x}_T$ ($q(\mathbf{x}_T \vert\mathbf{x}_{T-1})$) is nearly an [isotropic](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic#:~:text=TLDR%3A An isotropic gaussian is,Σ is the covariance matrix.) Gaussian distribution. Therefore if we manage to learn the reverse distribution $q(\mathbf{x}_{T-1} \vert\mathbf{x}_{T})$ , we can sample $\mathbf{x}_T$ from $\mathcal{N} (\mathbf{0}, \mathbf{I})$, run the reverse process and acquire a sample from $q(\mathbf{x}_0)$, generating a novel data point from the original data distribution.

The question is how we can model the reverse diffusion process.

### Approximating the reverse process with a neural network

In practical terms, we don't know $q(\mathbf{x}_{t-1} \vert\mathbf{x}_{t})$. It's intractable since statistical estimates of $q(\mathbf{x}_{t-1} \vert\mathbf{x}_{t})$ require computations involving the data distribution.

Instead, we approximate $q(\mathbf{x}_{t-1} \vert\mathbf{x}_{t})$ with a parameterized model $p_\theta$ (e.g. a neural network). Since $q(\mathbf{x}_{t-1} \vert\mathbf{x}_{t})$ will also be Gaussian, for small enough $\beta_t$, we can choose $p_\theta$ to be Gaussian and just parameterize the mean and variance:
$$
p_{\theta}(\mathbf{x}_{t-1} \vert\mathbf{x}_{t}) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t,t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)
$$

#### Function Inputs:

- **$ \boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t) $:**
  - **Inputs:**
    - $ \mathbf{x}_t $: The noisy data at timestep $ t $.
    - $ t $: The current timestep.
  - **Output:**
    - The predicted mean $ \boldsymbol{\mu}_{\theta} $ for the distribution of $ \mathbf{x}_{t-1} $.

- **$ \boldsymbol{\Sigma}_{\theta}(\mathbf{x}_t, t) $:**
  - **Inputs:**
    - $ \mathbf{x}_t $: The noisy data at timestep $ t $.
    - $ t $: The current timestep.
  - **Output:**
    - The predicted covariance $ \boldsymbol{\Sigma}_{\theta} $ for the distribution of $ \mathbf{x}_{t-1} $.

<img src="../../Library/Application Support/typora-user-images/截屏2024-10-02 下午6.54.04.png" alt="截屏2024-10-02 下午6.54.04" style="zoom:75%;" />

If we apply the reverse formula for all timesteps ($p(\mathbf{x}_{0:T})$, also called trajectory), we can go from $\mathbf{x}_T$ to the data distribution:
$$
p(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
$$
By additionally conditioning the model on timestep $t$, it will learn to predict the Gaussian parameters (meaning the mean $\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)$ and the covariance matrix $\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$ for each timestep.

Collectively, what this set of assumptions describes is a steady noisification of an image input over time; we progressively corrupt an image by adding Gaussian noise until eventually it becomes completely identical to pure Gaussian noise. Visually, this process is depicted in Figure 3. 

Note that our encoder distributions $q(\mathbf{x}_t|\mathbf{x}_{t−1})$ are no longer parameterized by $\phi$, as they are completely modeled as Gaussians with defined mean and variance parameters at each timestep. Therefore, in a VDM, we are only interested in learning conditionals $p_{\theta}(\mathbf{x}_{t-1} \vert\mathbf{x}_{t})$, so that we can simulate new data. After optimizing the VDM, the sampling procedure is as simple as sampling Gaussian noise from $p(\mathbf{x_T})$ and iteratively running
the denoising transitions $p_{\theta}(\mathbf{x}_{t-1} \vert\mathbf{x}_{t})$ for T steps to generate a novel $\mathbf{x}_0$.

VDM can be optimized by maximizing the ELBO, which can be derived as:
$$
\begin{align}
\log p(\mathbf{x}_0) &= \log \int p(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \tag{34}\\

&= \log \int \frac{p(\mathbf{x}_{0:T}) q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} d\mathbf{x}_{1:T} &\text{Note: }1=\frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} d\mathbf{x}_{1:T} \tag{35}\\

&= \log \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right] &\text{Convert into Expectation format}\tag{36}\\

&\geq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right] &\text{Apply Jensen's inequality}\tag{37}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{\prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] &\text{Subsitute } p(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)\text{ and } q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})\tag{38}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) p_\theta(\mathbf{x}_0|\mathbf{x}_1) \prod_{t=2}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_T|\mathbf{x}_{T-1}) \prod_{t=1}^{T-1} q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] \tag{39}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) p_\theta(\mathbf{x}_0|\mathbf{x}_1) \prod_{t=1}^{T-1} p_\theta(\mathbf{x}_{t}|\mathbf{x}_{t+1})}{q(\mathbf{x}_T|\mathbf{x}_{T-1}) \prod_{t=1}^{T-1} q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] &\text{Change numerator syntax to make it consistency with the denominator}\tag{40}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} + \log\frac{\prod_{t=1}^{T-1} p_\theta(\mathbf{x}_{t}|\mathbf{x}_{t+1})}{\prod_{t=1}^{T-1} q(\mathbf{x}_t|\mathbf{x}_{t-1})}\right] &\text{Log Property: log(a*b)=log(a)+log(b)}\\

&=\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] + \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \prod_{t=1}^{T-1} \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] &\text{Linearity of Expectation}\tag{41}\\

&=\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] 
+ \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] 
+ \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \sum_{t=1}^{T-1} \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] &\text{Decomposite the first term and convert the product into summation}\tag{42}\\

&=\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] 
+ \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] 
+ \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] \tag{43}\\

&=\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] 
+ \mathbb{E}_{q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] 
+ \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] \tag{44}\\\

&= \underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]}_{\text{reconstruction term}}
- \underbrace{\mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T)) \right]}_{\text{prior matching term}}- \underbrace{\sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ D_{\text{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \parallel p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})) \right]}_{\text{consistency term}}\tag{45}
\end{align}
$$


> ## Some interpretation about from equation 43 to equation 44
>
> 
>
> ### **Understanding Marginalization in Probability Theory**
>
> **Marginalization** is a fundamental concept in probability theory. It involves summing or integrating out variables from a joint probability distribution to obtain the marginal distribution of a subset of variables.
>
> #### **Simple Example:**
>
> Suppose we have two random variables, $ X $ and $ Y $, with a joint probability distribution $ p(X, Y) $. We can find the marginal distribution of $ X $ by integrating (if continuous) or summing (if discrete) over all possible values of $ Y $:
>
> $$
> p(X) = \int p(X, Y) \, dY \quad \text{(continuous case)}
> $$
> $$
> p(X) = \sum_Y p(X, Y) \quad \text{(discrete case)}
> $$
>
> This process **"marginalizes out"** the variable $ Y $, leaving us with the distribution of $ X $ alone.
>
> ---
>
> ### **Simplifying Expectations Using Marginalization**
>
> When calculating the expectation of a function that depends only on a subset of variables, we can marginalize out the other variables to simplify the computation.
>
> #### **Example with Expectation:**
>
> Suppose we want to compute:
>
> $$
> \mathbb{E}_{p(X, Y)}[f(X)] = \int \int f(X) \, p(X, Y) \, dY \, dX
> $$
> Since $ f(X) $ does not depend on $ Y $, we can rewrite the expectation as:
>
> $$
> \mathbb{E}_{p(X, Y)}[f(X)] = \int f(X) \left( \int p(X, Y) \, dY \right) dX = \int f(X) \, p(X) \, dX = \mathbb{E}_{p(X)}[f(X)]
> $$
> Here, we've marginalized out $ Y $ from the joint distribution $ p(X, Y) $ to obtain the marginal distribution $ p(X) $, simplifying the expectation.
>
> ---
>
> ### **Applying to the Diffusion Model's Loss Function**
>
> Now, let's relate this to your original question about the diffusion model's loss function and the transition from Equation (43) to Equation (44).
>
> #### **Equations Overview:**
>
> **Equation (43):**
> $$
> \begin{aligned}
> &\mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] \\
> &+ \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] \\
> &+ \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right]
> \end{aligned}
> $$
> **Equation (44):**
> $$
> \begin{aligned}
> &\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] \\
> &+ \mathbb{E}_{q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] \\
> &+ \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right]
> \end{aligned}
> $$
> **Goal:** Understand how to go from Equation (43) to Equation (44) by marginalizing out variables.
>
> ---
>
> ### **Step-by-Step Explanation with Concrete Examples**
>
> #### **First Term:**
>
> **Equation (43):**
>
> $$
> \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
> $$
>
> **Observation:** The function $ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) $ depends only on $ \mathbf{x}_0 $ and $ \mathbf{x}_1 $.
>
> **Simplification Process:**
>
> 1. **Identify Dependent Variables:** The function inside the expectation depends only on $ \mathbf{x}_0 $ and $ \mathbf{x}_1 $.
>
> 2. **Marginalize Out Other Variables:** Since the expectation is over $ q(\mathbf{x}_{1:T}|\mathbf{x}_0) $, but the function does not depend on $ \mathbf{x}_2, \dots, \mathbf{x}_T $, we can integrate out these variables.
>
> 3. **Compute Marginal Distribution:**
>    $$
>    q(\mathbf{x}_1|\mathbf{x}_0) = \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \, d\mathbf{x}_{2:T}
>    $$
>    
> 4. **Simplify Expectation:**
>
>    $$
>    \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] = \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
>    $$
>
> **Concrete Example:**
>
> Suppose $ \mathbf{x}_{1:T} $ are scalar variables for simplicity.
>
> - **Joint Distribution:** $ q(\mathbf{x}_{1:T}|\mathbf{x}_0) = q(\mathbf{x}_1|\mathbf{x}_0) \cdot q(\mathbf{x}_2|\mathbf{x}_1) \cdot \dots \cdot q(\mathbf{x}_T|\mathbf{x}_{T-1}) $
>
> - **Marginal Distribution:** To get $ q(\mathbf{x}_1|\mathbf{x}_0) $, we integrate over $ \mathbf{x}_2, \dots, \mathbf{x}_T $:
> $$
> q(\mathbf{x}_1|\mathbf{x}_0) = \int \dots \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \, d\mathbf{x}_2 \dots d\mathbf{x}_T
> $$
>   But since $ q $ is a Markov chain, and $ \mathbf{x}_1 $ depends only on $ \mathbf{x}_0 $, this simplifies to $ q(\mathbf{x}_1|\mathbf{x}_0) $.
>
> - **Simplified Expectation:**
>   $$
>   \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] = \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
>   $$
>
> #### **Second Term:**
>
> **Equation (43):**
> $$
> \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right]
> $$
> **Observation:** The function depends only on $ \mathbf{x}_{T-1} $ and $ \mathbf{x}_T $.
>
> **Simplification Process:**
>
> 1. **Identify Dependent Variables:** Function depends on $ \mathbf{x}_{T-1} $ and $ \mathbf{x}_T $.
>
> 2. **Marginalize Out Other Variables:** Integrate over $ \mathbf{x}_1, \dots, \mathbf{x}_{T-2} $.
>
> 3. **Compute Marginal Distribution:**
>    $$
>    q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0) = \int \dots \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \, d\mathbf{x}_1 \dots d\mathbf{x}_{T-2}
>    $$
>    
> 4. **Simplify Expectation:**
>    $$
>    \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] = \mathbb{E}_{q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right]
>    $$
>
> **Concrete Example:**
>
> - **Joint Distribution:** $ q(\mathbf{x}_{1:T}|\mathbf{x}_0) $ as before.
>
> - **Marginal Distribution:**
>   $$
>   q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0) = \int q(\mathbf{x}_1|\mathbf{x}_0) \cdot \dots \cdot q(\mathbf{x}_{T-2}|\mathbf{x}_{T-3}) \cdot q(\mathbf{x}_{T-1}|\mathbf{x}_{T-2}) \cdot q(\mathbf{x}_T|\mathbf{x}_{T-1}) \, d\mathbf{x}_1 \dots d\mathbf{x}_{T-2}
>   $$
>   Since $ \mathbf{x}_{T-1} $ depends only on $ \mathbf{x}_{T-2} $, and so on, integrating over $ \mathbf{x}_1, \dots, \mathbf{x}_{T-2} $ leaves us with the joint distribution of $ \mathbf{x}_{T-1} $ and $ \mathbf{x}_T $ conditioned on $ \mathbf{x}_0 $.
>
> #### **Third Term (Sum):**
>
> **Equation (43):**
> $$
> \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right]
> $$
> **Observation:** Each term in the sum depends on $ \mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1} $.
>
> **Simplification Process for Each Term:**
>
> 1. **Identify Dependent Variables:** Function depends on $ \mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1} $.
>
> 2. **Marginalize Out Other Variables:** Integrate over all $ \mathbf{x}_s $ where $ s \neq t-1, t, t+1 $.
>
> 3. **Compute Marginal Distribution:**
>    $$
>    q(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1}|\mathbf{x}_0) = \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \, d\mathbf{x}_{1:\,t-2} \, d\mathbf{x}_{t+2:\,T}
>    $$
>    
> 4. **Simplify Expectation:**
>    $$
>    \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] = \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right]
>    $$
>
> **Concrete Example:**
>
> For a specific $ t $, say $ t = 2 $:
>
> - **Marginal Distribution:**
>   $$
>   q(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3|\mathbf{x}_0) = \int q(\mathbf{x}_{1:T}|\mathbf{x}_0) \, d\mathbf{x}_{4}, \dots, d\mathbf{x}_{T}
>   $$
>   
> - **Simplified Expectation:**
>   $$
>   \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_2|\mathbf{x}_3)}{q(\mathbf{x}_2|\mathbf{x}_1)} \right] = \mathbb{E}_{q(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_2|\mathbf{x}_3)}{q(\mathbf{x}_2|\mathbf{x}_1)} \right]
>   $$
>
> ---
>
> ### **Key Takeaways:**
>
> 1. **Marginalization Simplifies Expectations:**
>
>    When the function inside the expectation depends only on a subset of variables, we can integrate out (marginalize) the other variables to simplify the expectation.
>
> 2. **Marginal Distributions:**
>
>    The marginal distribution of the variables of interest is obtained by integrating over the variables we are not interested in.
>
> 3. **Application to Diffusion Models:**
>
>    In the diffusion model's loss function, this process allows us to reduce the complexity of the expectations by focusing only on the relevant variables for each term.
>
> ---
>
> ### **Why This Works:**
>
> - **Properties of Expectations:**
>
>   The expectation of a function $ f(\mathbf{x}) $ with respect to a joint distribution $ q(\mathbf{x}) $ depends only on the variables that $ f $ depends on.
>
> - **Mathematical Justification:**
>
>   $$
>   \mathbb{E}_{q(\mathbf{x})}[f(\mathbf{x}_S)] = \int f(\mathbf{x}_S) \, q(\mathbf{x}) \, d\mathbf{x} = \int f(\mathbf{x}_S) \, q(\mathbf{x}_S) \, d\mathbf{x}_S
>   $$
>   Here, $ \mathbf{x}_S $ is the subset of variables that $ f $ depends on, and $ q(\mathbf{x}_S) $ is the marginal distribution of $ \mathbf{x}_S $.
>   
> - **Integration Over Irrelevant Variables:**
>
>   By integrating over variables that $ f $ does not depend on, we effectively remove them from the expectation calculation.
>



> ## Some interpretation about from equation 44 to equation 45
>
> Certainly! Let's break down how to derive Equation (45) from Equation (44) step by step, including how expectations of log ratios can be expressed as KL divergences.
>
> **Equation (44):**
>
> $$
> \begin{aligned}
> &= \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right] \\
> &\quad + \mathbb{E}_{q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] \\
> &\quad + \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right]
> \end{aligned}
> $$
> Our goal is to rewrite the second and third terms as expectations of negative KL divergences, leading to Equation (45):
>
> **Equation (45):**
>
> $$
> \begin{aligned}
> &= \underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]}_{\text{reconstruction term}} \\
> &\quad - \underbrace{\mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ D_{\text{KL}}\left(q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T)\right) \right]}_{\text{prior matching term}} \\
> &\quad - \underbrace{\sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ D_{\text{KL}}\left(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \parallel p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})\right) \right]}_{\text{consistency term}}
> \end{aligned}
> $$
> **Step 1: Understanding KL Divergence**
>
> The KL divergence between two distributions $ q(z) $ and $ p(z) $ is defined as:
>
> $$
> D_{\text{KL}}(q(z) \parallel p(z)) = \mathbb{E}_{q(z)} \left[ \log \frac{q(z)}{p(z)} \right]
> $$
> The negative KL divergence is:
>
> $$
> D_{\text{KL}}(q(z) \parallel p(z)) = \mathbb{E}_{q(z)} \left[ \log \frac{p(z)}{q(z)} \right]
> $$
> **Step 2: Rewriting the Second Term**
>
> **Second Term in Equation (44):**
>
> You start with the second term from Equation (44):
>
> $$
> \mathbb{E}_{q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right]
> $$
> **Step 1: Expanding the Expectation as an Integral**
>
> Express the expectation as a double integral:
>
> $$
> \int \int q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0) \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] d\mathbf{x}_{T-1} d\mathbf{x}_T
> $$
> **Step 2: Applying the Markov Property**
>
> Since the diffusion process is Markovian, you correctly factorize the joint distribution:
>
> $$
> q(\mathbf{x}_{T-1}, \mathbf{x}_T|\mathbf{x}_0) = q(\mathbf{x}_{T-1}|\mathbf{x}_0) \, q(\mathbf{x}_T|\mathbf{x}_{T-1})
> $$
> This is valid because, in a Markov process, the future state $ \mathbf{x}_T $ depends only on the current state $ \mathbf{x}_{T-1} $ and not directly on $ \mathbf{x}_0 $.
>
> **Step 3: Substituting the Factorized Distribution**
>
> You substitute this back into the integral:
>
> $$
> \int \int q(\mathbf{x}_{T-1}|\mathbf{x}_0) \, q(\mathbf{x}_T|\mathbf{x}_{T-1}) \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] d\mathbf{x}_{T-1} d\mathbf{x}_T
> $$
> **Step 4: Separating the Expectations**
>
> You correctly rewrite the double integral as nested expectations:
>
> $$
> \mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ \mathbb{E}_{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] \right]
> $$
> This separation is valid because you're integrating (or taking expectations) sequentially: first over $ \mathbf{x}_T $ given $ \mathbf{x}_{T-1} $, and then over $ \mathbf{x}_{T-1} $ given $ \mathbf{x}_0 $.
>
> **Step 5: Recognizing the Inner Expectation as a Negative KL Divergence**
>
> The inner expectation can be identified as the negative KL divergence between $ q(\mathbf{x}_T|\mathbf{x}_{T-1}) $ and $ p(\mathbf{x}_T) $:
>
> $$
> \mathbb{E}_{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] = - D_{\text{KL}} \left( q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T) \right)
> $$
> **Explanation:**
>
> - **KL Divergence Definition:**
>   $$
>   D_{\text{KL}}(q(z) \parallel p(z)) = \int q(z) \log \frac{q(z)}{p(z)} dz
>   $$
>   
> - **Negative KL Divergence:**
>   $$
>   D_{\text{KL}}(q(z) \parallel p(z)) = \int q(z) \log \frac{p(z)}{q(z)} dz = \mathbb{E}_{q(z)} \left[ \log \frac{p(z)}{q(z)} \right]
>   $$
>   
>
> Applying this to our case:
>
> - **Inner Expectation as Negative KL Divergence:**
>   $$
>   \mathbb{E}_{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \left[ \log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T|\mathbf{x}_{T-1})} \right] = - D_{\text{KL}} \left( q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T) \right)
>   $$
>
> **Step 6: Substituting Back into the Original Expression**
>
> Now, the second term becomes:
>
> $$
> \mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ - D_{\text{KL}} \left( q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T) \right) \right]
> $$
> Which simplifies to:
>
> $$
> \mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ D_{\text{KL}} \left( q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T) \right) \right]
> $$
> This matches the second term in Equation (45):
>
> $$
> \underbrace{\mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ D_{\text{KL}}\left(q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T)\right) \right]}_{\text{prior matching term}}
> $$
> **Step 3: Rewriting the Third Term**
>
> The third term in Equation (44) is:
>
> $$
> \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right]
> $$
> Again, we can separate the expectations:
>
> $$
> \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ \mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] \right]
> $$
> Since $q(\mathbf{x}_{t-1}, \mathbf{x}_t, \mathbf{x}_{t+1}|\mathbf{x}_0) = q(\mathbf{x}_{t-1}|\mathbf{x}_0) q(\mathbf{x}_{t}|\mathbf{x}_{t-1}) q(\mathbf{x}_{t+1}|\mathbf{x}_{t})$
>
> Since $ q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t|\mathbf{x}_{t-1}) $ (due to the Markov property), we recognize the inner expectation as:
> $$
> \mathbb{E}_{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \left[ \log \frac{p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})}{q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] = - D_{\text{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \parallel p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1}))
> $$
> Therefore, the third term becomes:
>
> $$
> \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ D_{\text{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \parallel p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})) \right]
> $$
> **Step 4: Combining the Terms**
>
> Now, we can combine all the terms:
>
> 1. **Reconstruction Term:**
> $$
>    \mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]
> $$
>    This term remains unchanged.
>
> 2. **Prior Matching Term:**
>
>    
> $$
>    \mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ D_{\text{KL}}(q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T)) \right]
> $$
>    
>
> 3. **Consistency Term:**
>    $$
>    \sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ D_{\text{KL}}(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \parallel p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})) \right]
>    $$
>    
>    
>
> **Conclusion:**
>
> By rewriting the expectations of log ratios as negative KL divergences, we successfully derive Equation (45) from Equation (44):
>
> $$
> \begin{aligned}
> &= \underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]}_{\text{reconstruction term}} \\
> &\quad - \underbrace{\mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ D_{\text{KL}}\left(q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T)\right) \right]}_{\text{prior matching term}} \\
> &\quad - \underbrace{\sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ D_{\text{KL}}\left(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \parallel p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})\right) \right]}_{\text{consistency term}}
> \end{aligned}
> $$
> **Key Points:**
>
> - **Expectation of Log Ratio as KL Divergence:** The expectation of the logarithm of the ratio of two distributions $ \log \frac{p(z)}{q(z)} $ under $ q(z) $ is the negative KL divergence between $ q(z) $ and $ p(z) $.
>
> - **Markov Property:** In diffusion models, the Markov property allows us to simplify $ q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) = q(\mathbf{x}_t|\mathbf{x}_{t-1}) $, which is crucial for separating the expectations.
>
> - **Rewriting Expectations:** By carefully separating the expectations over different variables and recognizing patterns matching the KL divergence, we can transform the terms accordingly.
>



The derived form of the ELBO can be interpreted in terms of its individual components:

1. $\underbrace{\mathbb{E}_{q(\mathbf{x}_1|\mathbf{x}_0)} \left[ \log p_\theta(\mathbf{x}_0|\mathbf{x}_1) \right]}_{\text{reconstruction term}}$ :  can be interpreted as a reconstruction term, predicting the log probability of the original data sample given the first-step latent. This term also appears in a vanilla VAE, and can be trained similarly.
2. $\underbrace{\mathbb{E}_{q(\mathbf{x}_{T-1}|\mathbf{x}_0)} \left[ D_{\text{KL}}\left(q(\mathbf{x}_T|\mathbf{x}_{T-1}) \parallel p(\mathbf{x}_T)\right) \right]}_{\text{prior matching term}}$ :  s a prior matching term; it is minimized when the final latent distribution matches the Gaussian prior. This term requires no optimization, as it has no trainable parameters; furthermore, as we have assumed a large enough T such that the final distribution is Gaussian, this term eﬀectively becomes zero.
3. $\underbrace{\sum_{t=1}^{T-1} \mathbb{E}_{q(\mathbf{x}_{t-1}, \mathbf{x}_{t+1}|\mathbf{x}_0)} \left[ D_{\text{KL}}\left(q(\mathbf{x}_t|\mathbf{x}_{t-1}) \parallel p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})\right) \right]}_{\text{consistency term}}$:  is a consistency term; it endeavors to make the distribution at $\mathbf{x}_t$ consistent, from both forward and backward processes. That is, a denoising step from a noisier image should match the corresponding noising step from a cleaner image, for every intermediate timestep; this is reflected mathematically by the KL Divergence. This term is minimized when we train $p_\theta(\mathbf{x}_t|\mathbf{x}_{t+1})$ to match the Gaussian distribution $q(\mathbf{x}_t|\mathbf{x}_{t-1})$.

![截屏2024-10-04 上午11.25.48](../../Library/Application Support/typora-user-images/截屏2024-10-04 上午11.25.48.png)

Under this derivation, all terms of the ELBO are computed as expectations, and can therefore be approximated using Monte Carlo estimates. However, actually optimizing the ELBO using the terms we just derived might be suboptimal; because the consistency term is computed as an expectation over two random variables $\{\mathbf{x}_{t−1},\mathbf{x}_{t+1}\}$ for every timestep, the variance of its Monte Carlo estimate could potentially be higher than a
term that is estimated using only one random variable per timestep. As it is computed by summing up $T−1$ consistency terms, the final estimated value of the ELBO may have high variance for large $T$ values.

Let us instead try to derive a form for our ELBO where each term is computed as an expectation over only one random variable at a time. The key insight is that we can rewrite encoder transitions as $q(\mathbf{x}_t|\mathbf{x}_{t-1}) = q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0)$ where the extra conditioning term is superfluous due to the Markov property. Then, according to Bayes rule, we can rewrite each transition as:
$$
q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0) = \frac{q(\mathbf{x}_t,\mathbf{x}_{t-1},\mathbf{x}_0)}{q(\mathbf{x}_{t-1},\mathbf{x}_0)} = \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_0) q(\mathbf{x}_{t}|\mathbf{x}_0)\cancel{q(\mathbf{x}_0)}}{q(\mathbf{x}_{t-1}|\mathbf{x}_0) \cancel{q(\mathbf{x}_0)}}=\frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_0) q(\mathbf{x}_{t}|\mathbf{x}_0)}{q(\mathbf{x}_{t-1}|\mathbf{x}_0) }
$$
Armed with this new equation, we can retry the derivation resuming from the ELBO 
$$
\begin{align}
\log p(\mathbf{x}_0) &= \log \int p(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \tag{34}\\

&= \log \int \frac{p(\mathbf{x}_{0:T}) q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} d\mathbf{x}_{1:T} &\text{Note: }1=\frac{q(\mathbf{x}_{1:T}|\mathbf{x}_0)}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} d\mathbf{x}_{1:T} \tag{35}\\

&= \log \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right] &\text{Convert into Expectation format}\tag{36}\\

&\geq \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \right] &\text{Apply Jensen's inequality}\tag{37}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_{t})}{\prod_{t=1}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1})} \right] &\text{Subsitute } p(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t)\text{ and } q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})\tag{38}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) p_\theta(\mathbf{x}_0|\mathbf{x}_1) \prod_{t=2}^{T} p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{q(\mathbf{x}_1|\mathbf{x}_{0}) \prod_{t=2}^{T} q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)} \right] &\text{Same what we did previously}\tag{50}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_{0})} +\log \prod^T_{t=2}\frac{ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{ q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0)} \right] &\text{Decomposite the Logrithm}\tag{51}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_{0})} +\log \prod^T_{t=2}\frac{ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{ \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_0) q(\mathbf{x}_{t}|\mathbf{x}_0)}{q(\mathbf{x}_{t-1}|\mathbf{x}_0) } } \right] &\text{Subsitute }q(\mathbf{x}_t|\mathbf{x}_{t-1},\mathbf{x}_0) =\frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_0) q(\mathbf{x}_{t}|\mathbf{x}_0)}{q(\mathbf{x}_{t-1}|\mathbf{x}_0) }\tag{52}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T}|\mathbf{x}_0)} \left[ \log \frac{p(\mathbf{x}_T) p_\theta(\mathbf{x}_0|\mathbf{x}_1)}{q(\mathbf{x}_1|\mathbf{x}_{0})} +\log \prod^T_{t=2}\frac{ p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)}{ \frac{q(\mathbf{x}_{t-1}|\mathbf{x}_{t},\mathbf{x}_0) \cancel{q(\mathbf{x}_{t}|\mathbf{x}_0)}}{\cancel{q(\mathbf{x}_{t-1}|\mathbf{x}_0)} } } \right] &\text{Just accept we can cancel out these two terms, currently no explaination}\tag{53}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ 
\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0 | \mathbf{x}_1)}{\cancel{q(\mathbf{x}_1 | \mathbf{x}_0)}} + 
\log \frac{\cancel{q(\mathbf{x}_1 | \mathbf{x}_0)}}{q(\mathbf{x}_T | \mathbf{x}_0)} +
\log \prod_{t=2}^T \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)}
\right] \tag{54}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ 
\log \frac{p(\mathbf{x}_T)p_\theta(\mathbf{x}_0 | \mathbf{x}_1)}{q(\mathbf{x}_T | \mathbf{x}_0)} + 
\sum_{t=2}^T \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)}
\right] &\text{Change the product to summation}\tag{55}\\

&= \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[\log p_\theta(\mathbf{x}_0 | \mathbf{x}_1)\right] 
+ \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ 
\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_0)} \right]
+ \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \left[ 
\log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)}
\right] \tag{56}\\

&= \mathbb{E}_{q(\mathbf{x}_1 | \mathbf{x}_0)} \left[\log p_\theta(\mathbf{x}_0 | \mathbf{x}_1)\right] 
+ \mathbb{E}_{q(\mathbf{x}_T | \mathbf{x}_0)} \left[ 
\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_0)} \right]
+ \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x}_t, \mathbf{x}_{t-1} | \mathbf{x}_0)} \left[ 
\log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)}
\right] \tag{57}\\

&
\underbrace{= \mathbb{E}_{q(\mathbf{x}_1 | \mathbf{x}_0)} \left[\log p_\theta(\mathbf{x}_0 | \mathbf{x}_1)\right]}_{\text{reconstruction term}}
\underbrace{- D_{KL}(q(\mathbf{x}_T | \mathbf{x}_0) \parallel p(\mathbf{x}_T))  }_{\text{prior matching term}} 
\underbrace{- \sum_{t=2}^T \mathbb{E}_{q(\mathbf{x}_t | \mathbf{x}_0)} \left[D_{KL}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t))\right]}_{\text{denoising matching term}} &q(\mathbf{x}_t, \mathbf{x}_{t-1} | \mathbf{x}_0) = q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)q(\mathbf{x}_t|\mathbf{x}_0)
\tag{58}
\end{align}
$$
We have therefore successfully derived an interpretation for the ELBO that can be estimated with lower variance, as each term is computed as an expectation of at most one random variable at a time. This formulation also has an elegant interpretation, which is revealed when inspecting each individual term:

- Reconstruction term: can be interpreted as a reconstruction term; like its analogue in the ELBO of a vanilla VAE, this term can be approximated and optimized using a Monte Carlo estimate.
- Prior matching term: represents how close the distribution of the final noisified input is to the standard Gaussian prior. It has no trainable parameters, and is also equal to zero under our assumptions.
- Denoising matching term:  We learn desired denoising transition step $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ as an approximation to tractable, ground-truth denoising transition step $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_{0})$. The $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_{0})$  transition step can act as a ground-truth signal, since it defines how to denoise a noisy image $\mathbf{x}_t$ with access to what the final, completely denoised image $\mathbf{x}_0$ should be. This term is therefore minimized when the two denoising steps match as closely as possible, as measured by their KL Divergence.

As a side note, one observes that in the process of both ELBO derivations (Equation 45 and Equation 58), only the Markov assumption is used; as a result these formulae will hold true for any arbitrary Markovian HVAE. Furthermore, when we set T = 1, both of the ELBO interpretations for a VDM exactly recreate the ELBO equation of a vanilla VAE, as written in VAE loss function.

In this derivation of the ELBO,th ebulk of the optimization cost once again lies in the summation term, which dominates the reconstruction term. Whereas each KL Divergence term $ \left[D_{KL}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t))\right]$ is diﬃcult to minimize for arbitrary posteriors in arbitrarily complex Markovian HVAEs due to the added complexity of simultaneously learning the encoder, in a VDM we can leverage the Gaussian transition
assumption to make optimization tractable. By Bayes rule, we have:
$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{ q(\mathbf{x}_{t}|\mathbf{x}_{t-1}, \mathbf{x}_0) q(\mathbf{x}_{t-1}|\mathbf{x}_0)} {q(\mathbf{x}_{t}|\mathbf{x}_0)}
$$
As we already know that $q(\mathbf{x}_{t} | \mathbf{x}_{t-1}, \mathbf{x}_0)  = q(\mathbf{x}_{t} | \mathbf{x}_{t-1}) =  \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha}_t \mathbf{x}_{t-1}, (1 - \alpha_t)\mathbf{I})$ from our assumption regarding encoder transitions (Previous paragraph), , what remains is deriving for the forms of $q(\mathbf{x}_{t-1}|\mathbf{x}_0)$ and $q(\mathbf{x}_t|\mathbf{x}_0)$. Fortunately, these are also made tractable by utilizing the fact that the encoder transitions of a VDM are linear Gaussian models. Recall that under the reparameterization trick, samples $\mathbf{x}_t \sim q(\mathbf{x}_t|\mathbf{x}_{t-1})$   can be
rewritten as what we derive in the forward diffusion. 

![截屏2024-10-06 下午11.13.33](../../Library/Application Support/typora-user-images/截屏2024-10-06 下午11.13.33.png)

> ## Short Summary about What have already derived 
>
> $q(\mathbf{x}_{t} | \mathbf{x}_{t-1}, \mathbf{x}_0)  = q(\mathbf{x}_{t} | \mathbf{x}_{t-1})= \sqrt{\alpha}_t \mathbf{x}_{t-1}+\sqrt{(1 - \alpha_t)}\mathbf{I} \quad \sim \mathcal{N}(\mathbf{x}_t; \sqrt{\alpha}_t \mathbf{x}_{t-1}, (1 - \alpha_t)\mathbf{I})$
>
> $q(\mathbf{x}_{t-1} \vert \mathbf{x}_{0}) = \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0} + \sqrt{(1 - \bar{\alpha}_{t-1}})\mathbf{I} \quad \sim \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_{0}, (1 - \bar{\alpha}_{t-1})\mathbf{I})$
>
> $q(\mathbf{x}_t \vert \mathbf{x}_0) = \sqrt{\bar{\alpha}_t} \mathbf{x}_0+\sqrt{ (1 - \bar{\alpha}_t)}\mathbf{I} \quad \sim \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$

Furthermore, in the forward diffusion process, we also attempt to derive the formula of how get $\mathbf{x}_t$ from $\mathbf{x}_0$, the Gaussian form of $q(\mathbf{x}_t|\mathbf{x}_0)$. This derivation can be modified to also yield the Gaussian parameterization describing $q(\mathbf{x}_{t-1}|\mathbf{x}_0)$. Now, knowing the forms of both$q(\mathbf{x}_{t-1}|\mathbf{x}_0)$ and $q(\mathbf{x}_t|\mathbf{x}_0)$, we can process to calculate the form of $q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0)$ by substituting into the Bayes rule expansion:
$$
\begin{align}

q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)

&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\

&= \frac{
\mathcal{N}(\mathbf{x}_t; \sqrt{\alpha_t} \, \mathbf{x}_{t-1}, (1 - \alpha_t) \mathbf{I}) \, 
\mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0, (1 - \bar{\alpha}_{t-1}) \mathbf{I})
}{
\mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})
}\\

&\propto \exp \left\{ 
- \left[ 
\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \, \mathbf{x}_{t-1})^2}{2(1 - \alpha_t)} 
+ \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0)^2}{2(1 - \bar{\alpha}_{t-1})}
- \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0)^2}{2(1 - \bar{\alpha}_t)}
\right] 
\right\} &\text{Subsitute with pdf of the Gaussian Distribution} \tag{59}\\

&=\exp \left\{ 
- \frac{1}{2} \left[ 
\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \, \mathbf{x}_{t-1})^2}{1 - \alpha_t} 
+ \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0)^2}{1 - \bar{\alpha}_{t-1}}
- \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0)^2}{1 - \bar{\alpha}_t}
\right] 
\right\} \tag{74}\\

&= \exp \left\{ 
- \frac{1}{2} \left[ 
\frac{(-2\sqrt{\alpha_t} \, \mathbf{x}_t \mathbf{x}_{t-1} + \alpha_t \mathbf{x}_t^2)}{1 - \alpha_t} 
+ \frac{(\mathbf{x}_{t-1}^2 - 2\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_{t-1} \mathbf{x}_0)}{1 - \bar{\alpha}_{t-1}} 
+ C(\mathbf{x}_t, \mathbf{x}_0)
\right] 
\right\} &\text{After decomposite the qudratic term, we only left with the term with } \mathbf{x}_{t-1} \text{ ignore, the reset term and attributed to the constant term} \tag{75}\\

&\propto \exp \left\{ 
- \frac{1}{2} \left[ 
\frac{-2\sqrt{\alpha_t} \, \mathbf{x}_t \mathbf{x}_{t-1}}{1 - \alpha_t} 
+ \frac{\alpha_t \mathbf{x}_{t-1}^2}{1 - \alpha_t} 
+ \frac{\mathbf{x}_{t-1}^2}{1 - \bar{\alpha}_{t-1}} 
- \frac{2\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_{t-1} \mathbf{x}_0}{1 - \bar{\alpha}_{t-1}} 
\right] 
\right\} &\text{Ignore the constant term}\tag{76}\\

&= \exp \left\{ 
- \frac{1}{2} \left[ 
\left( \frac{\alpha_t}{1 - \alpha_t} + \frac{1}{1 - \bar{\alpha}_{t-1}} \right) \mathbf{x}_{t-1}^2 
- 2 \left( \frac{\sqrt{\alpha_t} \, \mathbf{x}_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0}{1 - \bar{\alpha}_{t-1}} \right) \mathbf{x}_{t-1}
\right] 
\right\} \tag{77} \\ 

& = \exp \left\{ 
- \frac{1}{2} \left[ 
\frac{\alpha_t (1 - \bar{\alpha}_{t-1}) + (1 - \alpha_t)}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} \mathbf{x}_{t-1}^2 
- 2 \left( \frac{\sqrt{\alpha_t} \, \mathbf{x}_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0}{1 - \bar{\alpha}_{t-1}} \right) \mathbf{x}_{t-1}
\right] 
\right\}\tag{78} \\

&= \exp \left\{ 
- \frac{1}{2} \left[ 
\frac{\alpha_t - \bar{\alpha}_t + 1 - \alpha_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} \mathbf{x}_{t-1}^2 
- 2 \left( \frac{\sqrt{\alpha_t} \, \mathbf{x}_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0}{1 - \bar{\alpha}_{t-1}} \right) \mathbf{x}_{t-1}
\right] 
\right\} \tag{79} \\

&= \exp \left\{ 
- \frac{1}{2} \left[ 
\frac{1 - \bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} \mathbf{x}_{t-1}^2 
- 2 \left( \frac{\sqrt{\alpha_t} \, \mathbf{x}_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0}{1 - \bar{\alpha}_{t-1}} \right) \mathbf{x}_{t-1}
\right] 
\right\} \tag{80} \\

&= \exp \left\{ 
- \frac{1}{2} \left( \frac{1 - \bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} \right) \left[
\mathbf{x}_{t-1}^2 - 2 \left( \frac{\frac{\sqrt{\alpha_t} \, \mathbf{x}_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0}{1 - \bar{\alpha}_{t-1}}}{\frac{1 - \bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}} \right) \mathbf{x}_{t-1} 
\right] 
\right\} \tag{81} \\

&= \exp \left\{ 
- \frac{1}{2} \left( \frac{1 - \bar{\alpha}_t}{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})} \right) \left[
\mathbf{x}_{t-1}^2 - 2\frac{ \left( \frac{\sqrt{\alpha_t} \, \mathbf{x}_t}{1 - \alpha_t} + \frac{\sqrt{\bar{\alpha}_{t-1}} \, \mathbf{x}_0}{1 - \bar{\alpha}_{t-1}} \right) (1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \, \mathbf{x}_{t-1}
\right] 
\right\} \tag{82} \\

& = \exp \left\{ 
- \frac{1}{2} \left( \frac{1}{\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}} \right) \left[
\mathbf{x}_{t-1}^2 - 2 \left( \frac{ \sqrt{\alpha_t} (1 - \bar{\alpha}_{t-1}) \, \mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \, \mathbf{x}_0}{1 - \bar{\alpha}_t} \right) \mathbf{x}_{t-1}
\right] 
\right\} \tag{83} \\

& \propto \, \mathcal{N}\left(\mathbf{x}_{t-1}; \underbrace{\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \mathbf{x}_0}{1 - \bar{\alpha}_t}}_{\mu_q(\mathbf{x}_t, \mathbf{x}_0)}, \underbrace{\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \, \mathbf{I}}_{\Sigma_q(t)} \right) \tag{84}

\end{align}
$$
where in Equation 75, $C(\mathbf{x}_t,\mathbf{x}_0)$ is a constant term with respect to $\mathbf{x}_{t−1}$ computed as a combination of only $\mathbf{x}_t$, $\mathbf{x}_0$, and $\alpha$ values; this term is implicitly returned in Equation 84 to complete the square.

We have therefore shown that at each step, $\mathbf{x}_{t−1} \sim q(\mathbf{x}_{t−1} | \mathbf{x}_t, \mathbf{x}_0)$ is normally distributed, with mean $\mu_q(\mathbf{x}_t,\mathbf{x}_0)$ that is a function of $\mathbf{x}_t$ and $\mathbf{x}_0$, and variance $\Sigma_q(t)$ as a function of $\alpha$ coeﬃcients. These $\alpha$ coeﬃcients are known and fixed at each timestep; they are either set permanently when modeled as hyperparameters, or treated as the current inference output of a network that seeks to model them. Following Equation 84, we can rewrite our variance equation as $\Sigma_q(t) = \sigma^2_q(t) \mathbf{I}$, where:
$$
\sigma^2_q(t) =\frac{(1 - \alpha_t)(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}
$$
In order to match approximate denoising transition step $p_\theta(\mathbf{x}_{t−1}|\mathbf{x}_t)$ to ground-truth denoising transition step $q(\mathbf{x}_{t−1} | \mathbf{x}_t, \mathbf{x}_0)$ as closely as possible, we can also model it as a Gaussian. Furthermore, as all $\alpha$ terms are known to be frozen at each timestep, we can immediately construct the variance of the approximate denoising transition step to also be $\Sigma_q(t) = \sigma^2_q(t) \mathbf{I}$. We must parameterize its mean $\mu_\theta(\mathbf{x}_t,t)$ as a function
of $\mathbf{x}_t$, however, since $p_\theta(\mathbf{x}_{t−1}|\mathbf{x}_t)$ does not condition on $\mathbf{x}_0$.

Recall that the KL Divergence between two Gaussian distributions is:
$$
D_{KL}\left( \mathcal{N}(\mathbf{x}; \mu_{\mathbf{x}}, \Sigma_{\mathbf{x}}) \, \| \, \mathcal{N}(\mathbf{y}; \mu_{\mathbf{y}}, \Sigma_{\mathbf{y}}) \right) = \frac{1}{2} \left[ \log \frac{|\Sigma_{\mathbf{y}}|}{|\Sigma_{\mathbf{x}}|} - d + \text{tr}(\Sigma_{\mathbf{y}}^{-1} \Sigma_{\mathbf{x}}) + (\mu_{\mathbf{y}} - \mu_{\mathbf{x}})^T \Sigma_{\mathbf{y}}^{-1} (\mu_{\mathbf{y}} - \mu_{\mathbf{x}}) \right] \tag{86}
$$
In our case, where we can set the variances of the two Gaussians to match exactly, optimizing the KL Divergence term reduces to minimizing the diﬀerence between the means of the two distributions:
$$
\begin{align}
&\quad \quad\arg \min_{\theta} \, D_{KL}\left(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \, \| \, p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)\right) \\
&= \arg \min_{\theta} \, D_{KL}\left( \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q(t)) \, \| \, \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta, \boldsymbol{\Sigma}_q(t)) \right) \tag{87}\\

&= \arg \min_{\theta} \, \frac{1}{2} \left[ \log \frac{|\boldsymbol{\Sigma}_q(t)|}{|\boldsymbol{\Sigma}_q(t)|} - d + \text{tr}((\boldsymbol{\Sigma}_q(t))^{-1} \boldsymbol{\Sigma}_q(t)) + (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)^T (\boldsymbol{\Sigma}_q(t))^{-1} (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q) \right] \tag{88}\\

&= \arg \min_{\theta} \, \frac{1}{2} \left[ \log 1 - d + d + (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)^T (\boldsymbol{\Sigma}_q(t))^{-1} (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q) \right] \tag{89}\\

&= \arg \min_{\theta} \, \frac{1}{2} \left[ (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)^T (\boldsymbol{\Sigma}_q(t))^{-1} (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q) \right] \tag{90}\\

&=\arg \min_{\theta} \, \frac{1}{2} \left[ (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q)^T ( \sigma^2_q(t) \mathbf{I})^{-1} (\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q) \right] \tag{91}\\


&=\arg \min_{\theta} \, \frac{1}{2\sigma^2_q(t)}\left[||\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q||_2^2\right] \tag{92}\\


\end{align}
$$
where we have written $\boldsymbol{\mu}_q$ as shorthand for $\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0)$, and $\boldsymbol{\mu}_\theta$ as shorthand for $\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)$ for brevity. In other words, we want to optimize a $\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)$ that matches $\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0)$, which from our derived Equation 84, takes the form  (In MSML650 Final project, we use this formula for the inference step to calculate the mean):
$$
\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \mathbf{x}_0}{1 - \bar{\alpha}_t} \tag{93}
$$
As $\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)$ also conditions on $\mathbf{x}_t$, we can match $\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0)$ closely by setting it to the following form:
$$
\boldsymbol{\mu}_{\theta}(\mathbf{x}_t, t) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)}{1 - \bar{\alpha}_t} \tag{94}
$$
where $\hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)$ is parameterized by a neural network that seeks to predict $\mathbf{x}_0$ from noisy image $\mathbf{x}_t$ and time index t. Then, the optimization problem simplifies to:
$$
\begin{align}
&\quad \quad\arg \min_{\theta} \, D_{KL}\left(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \, \| \, p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)\right) \\
&= \arg \min_{\theta} \, D_{KL}\left( \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q(t)) \, \| \, \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta, \boldsymbol{\Sigma}_q(t)) \right) \tag{95}\\


&=\arg \min_{\theta} \, \frac{1}{2\sigma^2_q(t)}\left[||\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q||_2^2\right] \tag{96}\\

&= \arg \min_{\theta} \, \frac{1}{2 \sigma_q^2(t)} \left[\left\| \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \, \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)}{1 - \bar{\alpha}_t} - \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \, \mathbf{x}_0}{1 - \bar{\alpha}_t} \right\|_2^2\right]
&\text{Subsitute the Mean we derived previously} \tag{96}\\

&= \arg \min_{\theta} \, \frac{1}{2 \sigma_q^2(t)} \left[\left\| \frac{\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \, \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)}{1 - \bar{\alpha}_t} - \frac{\sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \, \mathbf{x}_0}{1 - \bar{\alpha}_t} \right\|_2^2\right] \tag{97}\\

&= \arg \min_{\theta} \, \frac{1}{2 \sigma_q^2(t)} \left[\left\| \frac{\sqrt{\bar{\alpha}_{t-1}}(1 - \alpha_t)}{1 - \bar{\alpha}_t} \left( \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t) - \mathbf{x}_0 \right) \right\|_2^2\right]\tag{98}\\

&= \arg \min_{\theta} \, \frac{1}{2 \sigma_q^2(t)} \frac{\bar{\alpha}_{t-1}(1 - \alpha_t)^2}{(1 - \bar{\alpha}_t)^2} \left[\left\| \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t) - \mathbf{x}_0 \right\|_2^2\right] \tag{99} \\


\end{align}
$$
Therefore, optimizing a VDM boils down to learning a neural network to predict the original ground truth image from an arbitrarily noisified version of it . Furthermore, minimizing the summation term of our derived ELBO objective (Equation 58) across all noise levels can be approximated by minimizing the expectation over all timesteps:
$$
\arg \min_{\theta} \, \mathbb{E}_{t \sim U\{2, T\}} \left[ \mathbb{E}_{q(\mathbf{x}_t | \mathbf{x}_0)} \left[ D_{KL}\left( q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \, \| \, p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) \right) \right] \right]
$$

## The ultimate interpretation of how the loss function make sense

As we previously proved, a Variational Diﬀusion Model can be trained by simply learning a neural network to predict the original natural image x0 from an arbitrary noised version $\mathbf{x}_t$ and its time index $t$. However, $\mathbf{x}_0$ has two other equivalent parameterizations, which leads to two further interpretations for a VDM. (We only record the one method which is the orthdorx method that the original author used)

Firstly, we can utilize the reparameterization trick. In our derivation of the form of $q(\mathbf{x}_t|\mathbf{x}_0)$, we can rearrange previous equation to show that:
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}\\
\mathbf{x}_0 = \frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}}_t \epsilon_0}{\sqrt{\bar{\alpha}}_t} \tag{115}
$$
Plugging this into our previously derived true denoising transition mean $\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0)$, we can rederive as:
$$
\begin{align}
\boldsymbol{\mu}_q(\mathbf{x}_t, \mathbf{x}_0) &= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \mathbf{x}_0}{1 - \bar{\alpha}_t} \tag{116}\\

&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t + \sqrt{\bar{\alpha}_{t-1}} (1 - \alpha_t) \frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}}_t \boldsymbol{\epsilon}_0}{\sqrt{\bar{\alpha}}_t}}{1 - \bar{\alpha}_t} &\text{Subsitute: }\mathbf{x}_0 = \frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}}_t \boldsymbol{\epsilon}_0}{\sqrt{\bar{\alpha}}_t} \tag{117}\\

&=\frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t +  (1 - \alpha_t) \frac{\mathbf{x}_t-\sqrt{1-\bar{\alpha}}_t \boldsymbol{\epsilon}_0}{\sqrt{\alpha_t}}}{1 - \bar{\alpha}_t} \tag{118}\\

&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1}) \mathbf{x}_t}{1 - \bar{\alpha}_t} + \frac{(1 - \bar{\alpha}_t) \mathbf{x}_t}{(1 - \bar{\alpha}_t) \sqrt{\alpha_t}} - \frac{(1 - \alpha_t) \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_0}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} \tag{119} \\

&= \left( \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} + \frac{1 - \alpha_t}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} \right) \mathbf{x}_t - \frac{(1 - \alpha_t) \sqrt{1 - \bar{\alpha}_t}}{(1 - \bar{\alpha}_t)\sqrt{\alpha_t}} \, \boldsymbol{\epsilon}_0 \tag{120} \\

&= \left( \frac{\alpha_t(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t) \sqrt{\alpha_t}} + \frac{1 - \alpha_t}{(1 - \bar{\alpha}_t) \sqrt{\alpha_t}} \right) \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\epsilon}_0 \tag{121}\\

&= \frac{\alpha_t - \bar{\alpha}_t + 1 - \alpha_t}{(1 - \bar{\alpha}_t) \sqrt{\alpha_t}} \, \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\epsilon}_0 \tag{122} \\

&= \frac{1 - \bar{\alpha}_t}{(1 - \bar{\alpha}_t) \sqrt{\alpha_t}} \, \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\epsilon}_0 \tag{123} \\

& = \frac{1}{ \sqrt{\alpha_t}} \, \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\epsilon}_0 \tag{124}
\end{align}
$$
Therefore, we can set our approximate denoising transition mean $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ as:
$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)= \frac{1}{ \sqrt{\alpha_t}} \, \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\hat{\epsilon}}_\theta(\mathbf{x}_t, t) \tag{125}
$$
and the corresponding optimization problem becomes:
$$
\begin{align}
&\quad \quad\arg \min_{\theta} \, D_{KL}\left(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \, \| \, p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)\right) \\
&= \arg \min_{\theta} \, D_{KL}\left( \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q(t)) \, \| \, \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta, \boldsymbol{\Sigma}_q(t)) \right) \tag{95}\\


&=\arg \min_{\theta} \, \frac{1}{2\sigma^2_q(t)}\left[||\boldsymbol{\mu}_\theta - \boldsymbol{\mu}_q||_2^2\right] \tag{96}\\

&=\arg \min_{\theta} \, \frac{1}{2\sigma^2_q(t)}\left[ \left\|\frac{1}{ \sqrt{\alpha_t}} \, \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\hat{\epsilon}}_\theta(\mathbf{x}_t, t) - \frac{1}{ \sqrt{\alpha_t}} \, \mathbf{x}_t + \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\epsilon}_0 \right\|_2^2\right] \tag{127}\\

&=\arg \min_{\theta} \, \frac{1}{2\sigma^2_q(t)}\left[ \left\|  \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\epsilon}_0 - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \boldsymbol{\hat{\epsilon}}_\theta(\mathbf{x}_t, t) \right\|_2^2\right]\tag{128} \\

&=\arg \min_{\theta} \, \frac{1}{2\sigma^2_q(t)}\left[ \left\|  \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t} \sqrt{\alpha_t}} \, \left(\boldsymbol{\epsilon}_0 -\boldsymbol{\hat{\epsilon}}_\theta(\mathbf{x}_t, t)\right) \right\|_2^2\right]\tag{129} \\

&= \arg \min_{\theta} \, \frac{1}{2 \sigma_q^2(t)} \frac{(1 - \alpha_t)^2}{(1 - \bar{\alpha}_t) \alpha_t} \left[ \left\| \boldsymbol{\epsilon}_0 - \hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \right\|_2^2 \right] \tag{130}

\end{align}
$$
Here, $\hat{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t)$ is a neural network that learns to predict the source noise $\epsilon_0 \sim \mathcal{N}(\epsilon; \mathbf{0}, \mathbf{I})$ that determines $\mathbf{x}_t$ from $\mathbf{x}_0$. We have therefore shown that learning a VDM by predicting the original image $\mathbf{x}_0$ is equivalent to learning to predict the noise; empirically, however, some works have found that predicting the noise resulted in better performance. 

![截屏2024-10-10 下午10.33.58](../../Library/Application Support/typora-user-images/截屏2024-10-10 下午10.33.58.png)

# Classifier Guidance and Classifier-free Guidance

## **1. Background: Denoising Diffusion Models**

**Denoising Diffusion Probabilistic Models (DDPMs)** are a class of generative models that learn to generate data by reversing a forward diffusion process that gradually adds noise to the data until it becomes pure noise. The model is trained to invert this process, starting from random noise and progressively denoising it to produce realistic samples.

- **Forward Process (Diffusion):** Adds Gaussian noise to data over multiple time steps.
- **Reverse Process (Denoising):** A neural network is trained to remove noise step by step.

In conditional generation, we want the generated data to satisfy certain conditions, such as matching a text prompt or a class label.

---

## **2. Classifier Guidance**

### **2.1. What Is Classifier Guidance?**

**Classifier guidance** is a technique that steers the generative process of a diffusion model towards samples that are more likely under a pre-trained classifier conditioned on the desired attribute (e.g., class labels or text prompts).

### **2.2. How Does It Work?**

- **Separate Classifier Model:** Train or use a pre-trained classifier $ p_\phi(y|\mathbf{x}_t) $ that predicts the desired attribute $ y $ given a noisy sample $ \mathbf{x}_t $ at time step $ t $.
- **Modify the Denoising Process:** Adjust the reverse diffusion process using gradients from the classifier to guide the generation towards the desired condition.

**Mathematically:**

- The denoising model predicts the noise $ \epsilon_\theta(\mathbf{x}_t, t) $.
- The classifier provides a gradient $ \nabla_{\mathbf{x}_t} \log p_\phi(y|\mathbf{x}_t) $.
- The adjusted noise prediction incorporates the classifier gradient:

  $$
  \tilde{\epsilon}_\theta(\mathbf{x}_t, t) = \epsilon_\theta(\mathbf{x}_t, t) - s \sigma_t \nabla_{\mathbf{x}_t} \log p_\phi(y|\mathbf{x}_t)
  $$

  - $ s $: Guidance scale (controls the strength of guidance).
  - $ \sigma_t $: Noise level at time $ t $.

### **2.3. Goals of Classifier Guidance**

- **Enhanced Conditional Sampling:** Generate samples that are more consistent with the desired condition $ y $.
- **Improved Sample Quality:** Utilize the classifier's knowledge to steer the generative process, leading to higher-quality outputs.

### **2.4. Advantages and Disadvantages**

- **Advantages:**
  - Provides explicit control over the generated content.
  - Can use pre-trained classifiers, leveraging existing models.

- **Disadvantages:**
  - Requires a separate classifier model trained on noisy data at each diffusion step.
  - Increases computational complexity and memory usage.
  - May introduce artifacts if the classifier is imperfect.

### **2.5. Relation to Diffusion Models**

- Classifier guidance modifies the reverse diffusion process by incorporating information from a classifier.
- It aligns the generative process with the desired conditional distribution $ p(\mathbf{x}_0|y) $.

---

## **3. Classifier-Free Guidance**

### **3.1. What Is Classifier-Free Guidance?**

**Classifier-free guidance** is an alternative technique that achieves conditional generation without requiring a separate classifier model. Instead, it trains the diffusion model itself to perform both conditional and unconditional generation, allowing it to guide itself during sampling.

### **3.2. How Does It Work?**

- **Unified Model:** The denoising model $ \epsilon_\theta(\mathbf{x}_t, t, y) $ is trained to accept an optional condition $ y $.
- **Training Procedure:**
  - Randomly drop the condition $ y $ during training with a certain probability (e.g., 10% of the time), effectively training the model to handle both conditional and unconditional cases.
- **Guidance During Sampling:**
  - Combine the conditional and unconditional predictions to guide the generation:

    $$
    \tilde{\epsilon}_\theta(\mathbf{x}_t, t) = (1 + w) \epsilon_\theta(\mathbf{x}_t, t, y) - w \epsilon_\theta(\mathbf{x}_t, t)
    $$

    - $ w $: Guidance scale (controls the strength of guidance).
    - $ \epsilon_\theta(\mathbf{x}_t, t, y) $: Conditional prediction.
    - $ \epsilon_\theta(\mathbf{x}_t, t) $: Unconditional prediction (condition $ y $ is omitted or masked).

### **3.3. Goals of Classifier-Free Guidance**

- **Simplify the Architecture:** Eliminate the need for a separate classifier model.
- **Enhance Efficiency:** Reduce computational and memory overhead.
- **Improve Sample Quality:** Achieve similar or better guidance compared to classifier guidance.

### **3.4. Advantages and Disadvantages**

- **Advantages:**
  - No separate classifier is needed; the denoising model serves both roles.
  - Simplifies training and inference pipelines.
  - Reduces computational resources compared to classifier guidance.
  - Avoids potential mismatch between the classifier and the diffusion model.

- **Disadvantages:**
  - Requires careful training to ensure the model learns both conditional and unconditional distributions effectively.
  - Guidance strength must be tuned to balance fidelity and diversity.

### **3.5. Relation to Diffusion Models**

- Classifier-free guidance directly incorporates guidance into the diffusion model by leveraging its ability to predict both conditional and unconditional denoising steps.
- It modifies the sampling process to emphasize the desired condition without external models.

---

## **4. Comparing Classifier Guidance and Classifier-Free Guidance**

### **4.1. Key Differences**

- **Separate Classifier vs. Unified Model:**
  - *Classifier Guidance:* Requires a separate classifier model trained on noisy data.
  - *Classifier-Free Guidance:* Integrates conditional and unconditional modeling into the denoising network.

- **Computational Complexity:**
  - *Classifier Guidance:* More computationally intensive due to the additional classifier.
  - *Classifier-Free Guidance:* More efficient, as it avoids extra models.

- **Implementation Complexity:**
  - *Classifier Guidance:* More complex training and inference pipelines.
  - *Classifier-Free Guidance:* Simpler implementation.

### **4.2. Goals Alignment**

Both methods aim to:

- **Improve Conditional Generation:** Generate samples that better match the desired condition.
- **Enhance Sample Quality:** Produce higher-quality and more coherent outputs.

### **4.3. Trade-offs**

- **Flexibility:**
  - *Classifier Guidance:* Can use different classifiers for various conditions.
  - *Classifier-Free Guidance:* Conditions are integrated into the model, which may be less flexible but more cohesive.

- **Performance:**
  - Empirically, classifier-free guidance often matches or exceeds the performance of classifier guidance in terms of sample quality.

---

## **5. How These Concepts Relate to Denoising Diffusion Models**

### **5.1. Conditioning in Diffusion Models**

- Diffusion models can be extended to conditional generation tasks by incorporating additional information (e.g., class labels, text prompts).
- Guidance techniques influence the reverse diffusion process to produce samples consistent with the conditions.

### **5.2. Controlling the Reverse Process**

- **Classifier Guidance:** Uses external gradients to adjust the denoising process at each step, effectively steering the generation towards the desired condition.
- **Classifier-Free Guidance:** Adjusts the denoising predictions internally by combining conditional and unconditional outputs, providing a form of self-guidance.

### **5.3. Importance in Applications**

- **Text-to-Image Generation:** Both techniques are crucial in models like Stable Diffusion, where generating images that match text prompts is essential.
- **Fine-Grained Control:** Guidance allows users to influence the generated content more precisely, leading to more useful and customizable outputs.

---

## **6. Practical Considerations**

### **6.1. Choosing a Guidance Method**

- **Classifier-Free Guidance** is generally preferred due to its simplicity and efficiency.
- It is widely used in state-of-the-art models like Stable Diffusion and DALL·E 2.

### **6.2. Tuning the Guidance Scale**

- The guidance scale $ s $ or $ w $ controls the strength of the guidance.
- **Higher Guidance Scale:**
  - Leads to outputs more closely matching the condition.
  - May reduce diversity and introduce artifacts if set too high.
- **Lower Guidance Scale:**
  - Preserves diversity but may produce less accurate conditioning.

### **6.3. Implementation Tips**

- **Training for Classifier-Free Guidance:**
  - Randomly drop conditions during training to enable the model to handle both conditional and unconditional inputs.
  - Use techniques like **conditioning dropout** to improve robustness.

- **Balancing Fidelity and Diversity:**
  - Experiment with different guidance scales to find the optimal balance for your application.
  - Consider using techniques like **sampling schedules** to adjust guidance strength dynamically during generation.

---

## **7. Summary**

- **Classifier Guidance:**
  - Uses a separate classifier to guide the diffusion model.
  - Incorporates external gradients to adjust the reverse process.
  - Provides explicit control but increases complexity and resource usage.

- **Classifier-Free Guidance:**
  - Integrates conditional and unconditional modeling into the diffusion model.
  - Adjusts predictions internally to guide generation.
  - Simplifies implementation and is more resource-efficient.

- **Relation to Diffusion Models:**
  - Both methods modify the reverse diffusion process to produce samples that align with desired conditions.
  - They enhance the capability of diffusion models to perform conditional generation tasks effectively.



### **Classifier Guidance Involves a Pre-trained Classifier**

- **Pre-trained Classifier:** When using **classifier guidance** in diffusion models, you indeed involve a **separately trained classifier**. This classifier $ p_\phi(y|\mathbf{x}_t) $ is trained to predict the desired condition $ y $ (e.g., class labels, text prompts) given the **noisy data** $ \mathbf{x}_t $ at various time steps $ t $ of the diffusion process.
  
- **Purpose of the Classifier:** The classifier provides gradients that indicate how to adjust the noisy sample $ \mathbf{x}_t $ to make it more likely to belong to the desired class $ y $.

### **Training Focuses on the Original Denoising Diffusion Model**

- **Training the Diffusion Model:** During training, you focus on training the original **denoising diffusion model** $ \epsilon_\theta(\mathbf{x}_t, t) $ to predict the noise added at each time step. The model learns to reverse the forward diffusion process by denoising the samples progressively.

- **Separate Classifier Training:** The classifier is trained **separately** from the diffusion model. It is trained on noisy samples $ \mathbf{x}_t $ to predict the condition $ y $.

### **How Classifier Guidance Works in Practice**

1. **Train the Denoising Diffusion Model:**

   - **Objective:** Learn to predict the noise $ \epsilon $ added to the data at each time step $ t $, enabling the model to denoise samples effectively.
   - **Loss Function:** Typically uses a mean squared error loss between the predicted noise and the true noise.

2. **Train the Classifier:**

   - **Dataset:** Use noisy data $ \mathbf{x}_t $ at various time steps $ t $ with corresponding labels $ y $.
   - **Objective:** Learn to predict $ y $ given $ \mathbf{x}_t $, even when $ \mathbf{x}_t $ is noisy.

3. **Sampling with Classifier Guidance:**

   - **Initialization:** Start with random noise $ \mathbf{x}_T $.
   - **Iterative Denoising:** For each time step $ t $ from $ T $ down to $ 1 $:
     - **Standard Denoising Step:** Use the diffusion model to predict the noise $ \epsilon_\theta(\mathbf{x}_t, t) $.
     - **Compute Classifier Gradient:** Calculate the gradient $ \nabla_{\mathbf{x}_t} \log p_\phi(y|\mathbf{x}_t) $ from the classifier.
     - **Adjust the Prediction:**
       
       $$
       \tilde{\epsilon}_\theta(\mathbf{x}_t, t) = \epsilon_\theta(\mathbf{x}_t, t) - s \sigma_t \nabla_{\mathbf{x}_t} \log p_\phi(y|\mathbf{x}_t)
       $$
       
       - **$ s $:** Guidance scale controlling the strength of the classifier's influence.
       - **$ \sigma_t $:** Noise level at time $ t $.
     - **Update Sample:** Use $ \tilde{\epsilon}_\theta(\mathbf{x}_t, t) $ to compute $ \mathbf{x}_{t-1} $.

### **Key Points to Understand**

- **Separate Training Processes:**
  - **Diffusion Model Training:** Focused on learning to denoise without considering the condition $ y $.
  - **Classifier Training:** Focused on learning to predict $ y $ from noisy data.

- **Guidance During Sampling Only:**
  - The classifier's gradients are used **only during the sampling phase**, not during the training of the diffusion model.
  - The diffusion model remains unchanged during sampling; it's the use of the classifier's gradients that steers the generation process.

- **Advantages of This Approach:**
  - **Flexibility:** You can swap out the classifier to guide the model toward different conditions without retraining the diffusion model.
  - **Control:** Provides a mechanism to influence the generated samples to match desired attributes.

### **Comparison with Classifier-Free Guidance**

- **Classifier Guidance:**
  - Requires a **separate, pre-trained classifier**.
  - Adds computational overhead during sampling due to the need to compute classifier gradients.
  - Offers explicit control over the generation process via the classifier.

- **Classifier-Free Guidance:**
  - **No separate classifier is needed**; the diffusion model is trained to handle both conditional and unconditional cases.
  - Simpler and more efficient, as it avoids the extra computational cost of a classifier.
  - Guidance is achieved by adjusting the model's own predictions.

### **Summary**

- **Yes,** when using **classifier guidance**, you involve a **pre-trained classifier** to guide the sampling process of the diffusion model towards the desired condition.
- **During training,** you focus on training the original **denoising diffusion model**, which learns to reverse the diffusion process by predicting and removing noise at each time step.
- The **pre-trained classifier** is used during the **sampling phase** to adjust the denoising process based on the condition $ y $, steering the generated samples to align with the desired attributes.

### **Practical Implications**

- **Training Workflow:**
  - **Step 1:** Train the denoising diffusion model on your dataset to perform unconditional generation.
  - **Step 2:** Train a classifier on noisy data to predict conditions or labels.
  - **Step 3:** Use the classifier's gradients during sampling to guide the diffusion model towards generating samples that satisfy the desired condition.

- **Considerations:**
  - **Computational Resources:** Classifier guidance increases computational requirements during sampling.
  - **Model Compatibility:** Ensure the classifier is compatible with the diffusion model in terms of input data (noise levels, preprocessing).
  - **Guidance Strength:** Experiment with the guidance scale $ s $ to balance adherence to the condition and sample diversity.

### **Example Use Case**

Suppose you have a diffusion model trained to generate images of animals, and you want to generate images of cats specifically:

- **Train a Classifier:** Train a classifier to predict animal classes (e.g., cat, dog, bird) from noisy images at various noise levels.
- **Sampling with Guidance:** During sampling, use the classifier's gradients to guide the diffusion model to generate images that the classifier would label as "cat."
- **Result:** The diffusion model generates images that are more likely to be recognized as cats by the classifier.

### **Final Thoughts**

- **Understanding Roles:**
  - **Diffusion Model:** Generates data by denoising, trained independently of conditions.
  - **Classifier:** Guides the generation during sampling, influencing the output towards desired attributes.

- **Flexibility and Control:**
  - Classifier guidance allows you to conditionally generate samples without altering the diffusion model's training process.
  - You can change the condition by using different classifiers or adjusting the guidance scale.
