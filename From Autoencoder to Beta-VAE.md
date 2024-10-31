# From Autoencoder to Beta-VAE

Autocoder is invented to reconstruct high-dimensional data using a neural network model with a narrow bottleneck layer in the middle (oops, this is probably not true for [Variational Autoencoder](https://lilianweng.github.io/posts/2018-08-12-vae/#vae-variational-autoencoder), and we will investigate it in details in later sections). A nice byproduct is dimension reduction: the bottleneck layer captures a compressed latent encoding. Such a low-dimensional representation can be used as en embedding vector in various applications (i.e. search), help data compression, or reveal the underlying data generative factors.

# Notation

| Symbol                                  | Mean                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| $\mathcal{D}$                           | The dataset $\mathcal{D} = \{ \mathbf{x}^{(1)}, \mathbf{x}^{(2)}, \dots, \mathbf{x}^{(n)} \}$, , contains $n$ data samples; $\vert\mathcal{D}\vert =n$. |
| $\mathbf{x}^{(i)}$                      | Each data point is a vector of dimensions, $\mathbf{x}^{(i)} = [x^{(i)}_1, x^{(i)}_2, \dots, x^{(i)}_d]$. |
| $\mathbf{x}$                            | One data sample from the dataset, $\mathbf{x} \in \mathcal{D}$. |
| $\mathbf{x}’$                           | The reconstructed version of $\mathbf{x}$.                   |
| $\tilde{\mathbf{x}}$                    | The corrupted version of $\mathbf{x}$.                       |
| $\mathbf{z}$                            | The compressed code learned in the bottleneck layer.         |
| $a_j^{(l)}$                             | The activation function for the $j$-th neuron in the $l$-th hidden layer. |
| $g_{\phi}(.)$                           | The **encoding** function parameterized by $\phi$.           |
| $f_{\theta}(.)$                         | The **decoding** function parameterized by $\theta$.         |
| $q_{\phi}(\mathbf{z}\vert\mathbf{x})$   | Estimated posterior probability function, also known as **probabilistic encoder**. |
| $p_{\theta}(\mathbf{x}\vert\mathbf{z})$ | Likelihood of generating true data sample given the latent code, also known as **probabilistic decoder**. |

# Autoencoder

**Autoencoder** is a neural network designed to learn an identity function in an unsupervised way to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation. The idea was originated in [the 1980s](https://en.wikipedia.org/wiki/Autoencoder), and later promoted by the seminal paper by [Hinton & Salakhutdinov, 2006](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.3788&rep=rep1&type=pdf).

It consists of two networks:

- *Encoder* network: It translates the original high-dimension input into the latent low-dimensional code. The input size is larger than the output size.
- *Decoder* network: The decoder network recovers the data from the code, likely with larger and larger output layers.

<img src="/Users/jiayuanshen/Library/Application Support/typora-user-images/截屏2024-07-31 上午1.03.12.png" alt="截屏2024-07-31 上午1.03.12" style="zoom:50%;" />

The encoder network essentially accomplishes the [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction), just like how we would use Principal Component Analysis (PCA) or Matrix Factorization (MF) for. In addition, the autoencoder is explicitly optimized for the data reconstruction from the code. A good intermediate representation not only can capture latent variables, but also benefits a full [decompression](https://ai.googleblog.com/2016/09/image-compression-with-neural-networks.html)process.

The model contains an encoder function $g(.)$ parameterized by $\phi$ and a decoder function $f(.)$ parameterized by $\theta$. The low-dimensional code learned for input in the bottleneck layer is $\mathbf{z} = g_\phi(\mathbf{x})$ and the reconstructed input is $\mathbf{x}’ = f_\theta(g_\phi(\mathbf{x}))$.

The parameters $(\theta, \phi)$ are learned together to output a reconstructed data sample same as the original input, $\mathbf{x} \approx f_\theta(g_\phi(\mathbf{x}))$ , or in other words, to learn an identity function. There are various metrics to quantify the difference between two vectors, such as cross entropy when the activation function is sigmoid, or as simple as MSE loss:
$$
L_\text{AE}(\theta, \phi) = \frac{1}{n}\sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\mathbf{x}^{(i)})))^2
$$

# Denoising Autoencoder

Since the autoencoder learns the identity function, we are facing the risk of “overfitting” when there are more network parameters than the number of data points.

To avoid overfitting and improve the robustness, **Denoising Autoencoder** (Vincent et al. 2008) proposed a modification to the basic autoencoder. The input is partially corrupted by adding noises to or masking some values of the input vector in a stochastic manner, $\tilde{\mathbf{x}} \sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$. Then the model is trained to recover the original input (note: not the corrupt one).
$$
\begin{aligned}
\tilde{\mathbf{x}}^{(i)} &\sim \mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}}^{(i)} \vert \mathbf{x}^{(i)})\\
L_\text{DAE}(\theta, \phi) &= \frac{1}{n} \sum_{i=1}^n (\mathbf{x}^{(i)} - f_\theta(g_\phi(\tilde{\mathbf{x}}^{(i)})))^2
\end{aligned}
$$
where $\mathcal{M}_\mathcal{D}$ defines the mapping from the true data samples to the noisy or corrupted ones.

<img src="/Users/jiayuanshen/Library/Application Support/typora-user-images/截屏2024-07-31 上午1.13.06.png" alt="截屏2024-07-31 上午1.13.06" style="zoom:50%;" />

This design is motivated by the fact that humans can easily recognize an object or a scene even the view is partially occluded or corrupted. To “repair” the partially destroyed input, the denoising autoencoder has to discover and capture relationship between dimensions of input in order to infer missing pieces.

For high dimensional input with high redundancy, like images, the model is likely to depend on evidence gathered from a combination of many input dimensions to recover the denoised version rather than to overfit one dimension. This builds up a good foundation for learning *robust* latent representation.

The noise is controlled by a stochastic mapping $\mathcal{M}_\mathcal{D}(\tilde{\mathbf{x}} \vert \mathbf{x})$, and it is not specific to a particular type of corruption process (i.e. masking noise, Gaussian noise, salt-and-pepper noise, etc.). Naturally the corruption process can be equipped with prior knowledge

In the experiment of the original DAE paper, the noise is applied in this way: a fixed proportion of input dimensions are selected at random and their values are forced to 0. Sounds a lot like dropout, right? Well, the denoising autoencoder was proposed in 2008, 4 years before the dropout paper ([Hinton, et al. 2012](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)) ;)



# Introduction to Autoencoders

Autoencoders are an unsupervised learning technique in which we leverage neural networks for the task of **representation learning**. Specifically, we'll design a neural network architecture such that we *impose a bottleneck in the network which forces a **compressed** knowledge representation of the original input*. If the input features were each independent of one another, this compression and subsequent reconstruction would be a very difficult task. However, if some sort of structure exists in the data (ie. correlations between input features), this structure can be learned and consequently leveraged when forcing the input through the network's bottleneck.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-13 上午2.49.02.png" alt="截屏2024-09-13 上午2.49.02" style="zoom:50%;" />

As visualized above, we can take an unlabeled dataset and frame it as a supervised learning problem tasked with outputting $\hat x$, a **reconstruction of the original input $x$** . This network can be trained by minimizing the *reconstruction error*, ${\cal L}\left( {x,\hat x} \right)$, which measures the differences between our original input and the consequent reconstruction. The bottleneck is a key attribute of our network design; without the presence of an information bottleneck, our network could easily learn to simply memorize the input values by passing these values along through the network (visualized below).

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-13 上午3.32.17.png" alt="截屏2024-09-13 上午3.32.17" style="zoom:50%;" />

A bottleneck constrains the amount of information that can traverse the full network, forcing a learned compression of the input data.

*Note: In fact, if we were to construct a linear network (ie. without the use of nonlinear [activation functions](https://www.jeremyjordan.me/neural-networks-activation-functions/) at each layer) we would observe a similar dimensionality reduction as observed in PCA. [See Geoffrey Hinton's discussion of this here.](https://www.coursera.org/learn/neural-networks/lecture/JiT1i/from-pca-to-autoencoders-5-mins?ref=jeremyjordan.me)*

The ideal autoencoder model balances the following:

- Sensitive to the inputs enough to accurately build a reconstruction.
- Insensitive enough to the inputs that the model doesn't simply memorize or overfit the training data.

This trade-off forces the model to maintain only the variations in the data required to reconstruct the input without holding on to redundancies within the input. For most cases, this involves constructing a loss function where one term encourages our model to be sensitive to the inputs (ie. reconstruction loss ${\cal L}\left( {x,\hat x} \right)$) and a second term discourages memorization/overfitting (ie. an added regularizer).
$$
{\cal L}\left( {x,\hat x} \right) + regularizer
$$
We'll typically add a scaling parameter in front of the regularization term so that we can adjust the trade-off between the two objectives.

In this post, I'll discuss some of the standard autoencoder architectures for imposing these two constraints and tuning the trade-off; in a follow-up post I'll discuss [variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/) which builds on the concepts discussed here to provide a more powerful model.

#### Undercomplete autoencoder

The simplest architecture for constructing an autoencoder is to constrain the number of nodes present in the hidden layer(s) of the network, limiting the amount of information that can flow through the network. By penalizing the network according to the reconstruction error, our model can learn the most important attributes of the input data and how to best reconstruct the original input from an "encoded" state. Ideally, this encoding will **learn and describe latent attributes of the input data**.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-13 上午4.46.52.png" alt="截屏2024-09-13 上午4.46.52" style="zoom:50%;" />

Because neural networks are capable of learning nonlinear relationships, this can be thought of as a more powerful (nonlinear) generalization of PCA. Whereas PCA attempts to discover a lower dimensional hyperplane which describes the original data, autoencoders are capable of learning nonlinear manifolds (a manifold is defined in simple terms as a continuous, non-intersecting surface). The difference between these two approaches is visualized below.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-13 上午4.48.17.png" alt="截屏2024-09-13 上午4.48.17" style="zoom:50%;" />

For higher dimensional data, autoencoders are capable of learning a complex representation of the data (manifold) which can be used to describe observations in a lower dimensionality and correspondingly decoded into the original input space.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-13 上午5.13.30.png" alt="截屏2024-09-13 上午5.13.30" style="zoom:50%;" />

An undercomplete autoencoder has no explicit regularization term - we simply train our model according to the reconstruction loss. Thus, our only way to ensure that the model isn't memorizing the input data is the ensure that we've sufficiently restricted the number of nodes in the hidden layer(s).

For deep autoencoders, we must also be aware of the *capacity* of our encoder and decoder models. Even if the "bottleneck layer" is only one hidden node, it's still possible for our model to memorize the training data provided that the encoder and decoder models have sufficient capability to learn some arbitrary function which can map the data to an index.

Given the fact that we'd like our model to discover latent attributes within our data, it's important to ensure that the autoencoder model is not simply learning an efficient way to memorize the training data. Similar to supervised learning problems, we can employ various forms of regularization to the network in order to encourage good generalization properties; these techniques are discussed below.

#### Sparse autoencoders

Sparse autoencoders offer us an alternative method for introducing an information bottleneck without *requiring* a reduction in the number of nodes at our hidden layers. Rather, we'll construct our loss function such that we penalize *activations* within a layer. For any given observation, we'll encourage our network to learn an encoding and decoding which only relies on activating a small number of neurons. It's worth noting that this is a different approach towards regularization, as we normally regularize the *weights* of a network, not the activations.

A generic sparse autoencoder is visualized below where the opacity of a node corresponds with the level of activation. It's important to note that the individual nodes of a trained model which activate are *data-dependent*, different inputs will result in activations of different nodes through the network.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-13 上午5.38.33.png" alt="截屏2024-09-13 上午5.38.33" style="zoom:50%;" />

One result of this fact is that **we allow our network to sensitize individual hidden layer nodes toward specific attributes of the input data**. Whereas an undercomplete autoencoder will use the entire network for every observation, a sparse autoencoder will be forced to selectively activate regions of the network depending on the input data. As a result, we've limited the network's capacity to memorize the input data without limiting the networks capability to extract features from the data. This allows us to consider the latent state representation and regularization of the network *separately*, such that we can choose a latent state representation (ie. encoding dimensionality) in accordance with what makes sense given the context of the data while imposing regularization by the sparsity constraint.

There are two main ways by which we can impose this sparsity constraint; both involve measuring the hidden layer activations for each training batch and adding some term to the loss function in order to penalize excessive activations. These terms are:

- **L1 Regularization**: We can add a term to our loss function that penalizes the absolute value of the vector of activations $a$ in layer $h$ for observation $i$, scaled by a tuning parameter $\lambda$
  $$
  {\cal L}\left( {x,\hat x} \right) +  \lambda \sum\limits_i {\left| {a_i^{\left( h \right)}} \right|}
  $$

- **KL-Divergence**: In essence, KL-divergence is a measure of the difference between two probability distributions. We can define a sparsity parameter $\rho$ which denotes the average activation of a neuron over a collection of samples. This expectation can be calculated as where the subscript denotes the specific neuron in layer , summing the activations for training observations denoted individually as ${{\hat \rho }_ j} = \frac{1}{m}\sum\limits_{i} {\left[ {a_i^{\left( h \right)}\left( x \right)} \right]}$. In essence, by constraining the average activation of a neuron over a collection of samples we're encouraging neurons to only fire for a subset of the observations. We can describe as a Bernoulli random variable distribution such that we can leverage the KL divergence (expanded below) to compare the ideal distribution $\rho$ to the observed distributions over all hidden layer nodes $\hat \rho$.

  *Note: A [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoullidistribution?ref=jeremyjordan.me) is "the probability distribution of a random variable which takes the value 1 with probability $p$ and the value 0 with probability $q=1-p$". This corresponds quite well with establishing the probability a neuron will fire.*

  The KL divergence between two Bernoulli distributions can be written as $\sum\limits_{j = 1}^{{l^{\left( h \right)}}} {\rho \log \frac{\rho }{{{{\hat \rho }_ j}}}}  + \left( {1 - \rho } \right)\log \frac{{1 - \rho }}{{1 - {{\hat \rho }_ j}}}$. This loss term is visualized below for an ideal distribution of $\rho = 0.2$, corresponding with the minimum (zero) penalty at this point.

  <img src="../../Library/Application Support/typora-user-images/截屏2024-09-15 下午7.09.41.png" alt="截屏2024-09-15 下午7.09.41" style="zoom:50%;" />

​	For this graph the x-axis is "Average activation of hidden unit $\hat \rho$", and the y-axis is "KL-divergence"

> **An Example:**
>
> Certainly! I'd be happy to help clarify these concepts for you. Let's break down the role of the KL-divergence in sparse autoencoders and then create some mock data to illustrate how the parameters $ \rho $ and $ \hat{\rho} $ function within this context.
>
> ### **Understanding Sparse Autoencoders and KL-Divergence**
>
> **Sparse Autoencoders:**
>
> A sparse autoencoder is a type of neural network used for unsupervised learning, where the goal is to learn efficient representations (encodings) of data by training the network to reconstruct its input. Sparsity is enforced to ensure that the learned representations capture essential features without redundancy.
>
> **Sparsity Constraint:**
>
> - **$ \rho $:** The sparsity parameter (a small value, e.g., 0.05) representing the desired average activation of each hidden neuron.
> - **$ \hat{\rho}_j $:** The actual average activation of hidden neuron $ j $ over the training dataset.
>
> **KL-Divergence in Sparse Autoencoders:**
>
> The Kullback-Leibler (KL) divergence measures how one probability distribution diverges from a second, expected probability distribution. In sparse autoencoders, we use KL-divergence to quantify the difference between the desired sparsity $ \rho $ and the actual average activations $ \hat{\rho}_j $ of the neurons.
>
> The sparsity penalty added to the loss function is:
>
> $
> \sum_{j=1}^{n_{\text{hidden}}} \text{KL}(\rho \,||\, \hat{\rho}_j) = \sum_{j=1}^{n_{\text{hidden}}} \left[ \rho \ln \left( \frac{\rho}{\hat{\rho}_j} \right) + (1 - \rho) \ln \left( \frac{1 - \rho}{1 - \hat{\rho}_j} \right) \right]
> $
>
> This penalty encourages the network to adjust weights such that $ \hat{\rho}_j $ approaches $ \rho $ for all hidden neurons.
>
> ### **Mock Data Illustration**
>
> Let's create a simple example with mock data to illustrate how $ \rho $ and $ \hat{\rho}_j $ function in a sparse autoencoder.
>
> #### **Setup:**
>
> - **Training Samples ($ m $)**: 5 input samples.
> - **Hidden Neurons ($ n_{\text{hidden}} $)**: 3 neurons in the hidden layer.
> - **Desired Sparsity ($ \rho $)**: 0.2 (we desire that each neuron is active 20% of the time).
>
> #### **Step 1: Compute Activations**
>
> Assume we have the following activations for each hidden neuron $ j $ when processing each sample $ i $:
>
> | Sample $ i $ | Neuron 1 ($ a_{i1} $) | Neuron 2 ($ a_{i2} $) | Neuron 3 ($ a_{i3} $) |
> | ------------ | --------------------- | --------------------- | --------------------- |
> | 1            | 0.9                   | 0.1                   | 0.05                  |
> | 2            | 0.8                   | 0.15                  | 0.1                   |
> | 3            | 0.85                  | 0.05                  | 0.2                   |
> | 4            | 0.9                   | 0.1                   | 0.15                  |
> | 5            | 0.88                  | 0.08                  | 0.05                  |
>
> These activations are outputs of the sigmoid function, hence they range between 0 and 1.
>
> #### **Step 2: Compute Average Activations ($ \hat{\rho}_j $)**
>
> Calculate $ \hat{\rho}_j $ for each neuron $ j $:
>
> $
> \hat{\rho}_j = \frac{1}{m} \sum_{i=1}^{m} a_{ij}
> $
>
> - **Neuron 1 ($ \hat{\rho}_1 $)**:
>
>   $
>   \hat{\rho}_1 = \frac{1}{5} (0.9 + 0.8 + 0.85 + 0.9 + 0.88) = \frac{4.33}{5} = 0.866
>   $
>
> - **Neuron 2 ($ \hat{\rho}_2 $)**:
>
>   $
>   \hat{\rho}_2 = \frac{1}{5} (0.1 + 0.15 + 0.05 + 0.1 + 0.08) = \frac{0.48}{5} = 0.096
>   $
>
> - **Neuron 3 ($ \hat{\rho}_3 $)**:
>
>   $
>   \hat{\rho}_3 = \frac{1}{5} (0.05 + 0.1 + 0.2 + 0.15 + 0.05) = \frac{0.55}{5} = 0.11
>   $
>
> #### **Step 3: Interpret $ \hat{\rho}_j $ and $ \rho $**
>
> - **Desired Sparsity ($ \rho = 0.2 $)**:
>
>   We want each neuron to have an average activation close to 0.2.
>
> - **Actual Average Activations ($ \hat{\rho}_j $)**:
>
>   - **Neuron 1 ($ \hat{\rho}_1 = 0.866 $)**: This neuron is highly active (86.6% on average), exceeding the desired sparsity.
>   - **Neuron 2 ($ \hat{\rho}_2 = 0.096 $)**: This neuron is less active than desired.
>   - **Neuron 3 ($ \hat{\rho}_3 = 0.11 $)**: This neuron is also less active than desired.
>
> #### **Step 4: Compute KL-Divergence Penalty**
>
> Compute the KL-divergence between $ \rho $ and $ \hat{\rho}_j $ for each neuron:
>
> $
> \text{KL}(\rho \,||\, \hat{\rho}_j) = \rho \ln \left( \frac{\rho}{\hat{\rho}_j} \right) + (1 - \rho) \ln \left( \frac{1 - \rho}{1 - \hat{\rho}_j} \right)
> $
>
> - **Neuron 1 ($ \hat{\rho}_1 = 0.866 $)**:
>
>   $
>   \begin{align*}
>   \text{KL}(\rho \,||\, \hat{\rho}_1) &= 0.2 \ln \left( \frac{0.2}{0.866} \right) + 0.8 \ln \left( \frac{0.8}{1 - 0.866} \right) \\
>   &= 0.2 \ln (0.2309) + 0.8 \ln (5.9701) \\
>   &= 0.2 \times (-1.4663) + 0.8 \times (1.7865) \\
>   &= -0.2933 + 1.4292 = 1.1359
>   \end{align*}
>   $
>
> - **Neuron 2 ($ \hat{\rho}_2 = 0.096 $)**:
>
>   $
>   \begin{align*}
>   \text{KL}(\rho \,||\, \hat{\rho}_2) &= 0.2 \ln \left( \frac{0.2}{0.096} \right) + 0.8 \ln \left( \frac{0.8}{0.904} \right) \\
>   &= 0.2 \ln (2.0833) + 0.8 \ln (0.8849) \\
>   &= 0.2 \times (0.7332) + 0.8 \times (-0.1223) \\
>   &= 0.1466 - 0.0978 = 0.0488
>   \end{align*}
>   $
>
> - **Neuron 3 ($ \hat{\rho}_3 = 0.11 $)**:
>
>   $
>   \begin{align*}
>   \text{KL}(\rho \,||\, \hat{\rho}_3) &= 0.2 \ln \left( \frac{0.2}{0.11} \right) + 0.8 \ln \left( \frac{0.8}{0.89} \right) \\
>   &= 0.2 \ln (1.8182) + 0.8 \ln (0.8989) \\
>   &= 0.2 \times (0.5978) + 0.8 \times (-0.1064) \\
>   &= 0.1196 - 0.0851 = 0.0345
>   \end{align*}
>   $
>
> #### **Step 5: Interpret KL-Divergence Values**
>
> - **Neuron 1 ($ \text{KL} = 1.1359 $)**:
>
>   High KL-divergence indicates a large discrepancy between the desired and actual activations. The neuron is overactive compared to the desired sparsity.
>
> - **Neuron 2 ($ \text{KL} = 0.0488 $)**:
>
>   Low KL-divergence suggests that the neuron's activity is close to the desired sparsity.
>
> - **Neuron 3 ($ \text{KL} = 0.0345 $)**:
>
>   Similar to Neuron 2, the neuron's activity is close to the desired sparsity.
>
> #### **Step 6: Applying Penalty in Training**
>
> The KL-divergence penalties are added to the overall loss function:
>
> $
> \text{Total Loss} = \text{Reconstruction Loss} + \beta \sum_{j=1}^{n_{\text{hidden}}} \text{KL}(\rho \,||\, \hat{\rho}_j)
> $
>
> - **$ \beta $** is a weight that balances the importance of the sparsity penalty relative to the reconstruction loss.
>
> - **Effect on Training**:
>
>   - The high penalty for Neuron 1 will push the network to adjust the weights connected to Neuron 1 to reduce its average activation.
>   - Neurons 2 and 3 receive smaller penalties, so their weights will be adjusted less aggressively.
>
> ### **Connecting to Variational Autoencoders (VAEs)**
>
> While sparse autoencoders and VAEs are different models, understanding the role of KL-divergence in sparse autoencoders can help in grasping similar concepts in VAEs.
>
> **Variational Autoencoders:**
>
> - VAEs are generative models that encode input data into a latent space characterized by probability distributions (usually Gaussian).
> - They use KL-divergence to measure the difference between the learned latent distribution $ q(z|x) $ and a prior distribution $ p(z) $ (often standard normal).
> - The KL-divergence acts as a regularizer, ensuring that the learned latent representations are close to the prior, which enables generative capabilities.
>
> **Similarity in Concepts:**
>
> - **Probability Distributions:** Both models use KL-divergence to compare distributions—Bernoulli distributions in sparse autoencoders and Gaussian distributions in VAEs.
> - **Regularization:** In both cases, KL-divergence serves as a penalty term that regularizes the network to produce desired statistical properties in activations or latent variables.
> - **Optimization Goal:** The goal is to minimize the loss function, which includes reconstruction loss and the KL-divergence penalty.
>
> ### **Summary**
>
> - **$ \rho $:** Desired average activation (sparsity level).
> - **$ \hat{\rho}_j $:** Actual average activation of neuron $ j $ over the dataset.
> - **KL-Divergence:** Measures how much $ \hat{\rho}_j $ diverges from $ \rho $ for each neuron.
> - **Sparsity Penalty:** Encourages neurons to have average activations close to $ \rho $, promoting sparsity.
> - **Training Effect:** The network adjusts weights to minimize the total loss, balancing reconstruction accuracy and sparsity.
>
> ### **Key Takeaways**
>
> - By viewing neuron activations as probabilities (of being "active" or "inactive"), we can model them using Bernoulli distributions.
> - The KL-divergence quantifies the discrepancy between the desired sparsity level and the actual activations.
> - This approach allows the network to learn sparse representations by penalizing neurons that are too active or not active enough.
>

> **Another Example about how to derive the KL-divergence**
>
> Certainly! I'd be happy to derive the formula for the Kullback-Leibler (KL) divergence between two Bernoulli distributions, which is used in the context of sparse autoencoders to enforce sparsity.
>
> ### **Overview**
>
> We will:
>
> 1. **Recall the Definition of KL-Divergence**
> 2. **Apply It to Bernoulli Distributions**
> 3. **Derive the Specific Formula**
> 4. **Explain Each Step in Detail**
>
> ---
>
> ### **1. Definition of KL-Divergence**
>
> The KL-divergence from a probability distribution $ P $ to a probability distribution $ Q $ is defined as:
>
> $
> D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \ln \left( \frac{P(x)}{Q(x)} \right)
> $
>
> This measures how one probability distribution $ P $ diverges from a second expected probability distribution $ Q $.
>
> ---
>
> ### **2. Bernoulli Distributions**
>
> A Bernoulli distribution is a discrete probability distribution with only two possible outcomes: 0 and 1 (e.g., "failure" and "success"). It's defined by a single parameter:
>
> - **For Distribution $ P $:** Success probability $ p $
> - **For Distribution $ Q $:** Success probability $ q $
>
> The probability mass functions (PMFs) are:
>
> - **For $ P $:**
>
>   $
>   P(x) = 
>   \begin{cases} 
>   p & \text{if } x = 1 \\
>   1 - p & \text{if } x = 0 
>   \end{cases}
>   $
>
> - **For $ Q $:**
>
>   $
>   Q(x) = 
>   \begin{cases} 
>   q & \text{if } x = 1 \\
>   1 - q & \text{if } x = 0 
>   \end{cases}
>   $
>
> ---
>
> ### **3. Derivation of the KL-Divergence Between Two Bernoulli Distributions**
>
> We can compute the KL-divergence $ D_{\text{KL}}(P \parallel Q) $ by summing over the possible values of $ x $ (0 and 1):
>
> $
> D_{\text{KL}}(P \parallel Q) = \sum_{x \in \{0,1\}} P(x) \ln \left( \frac{P(x)}{Q(x)} \right)
> $
>
> Let's compute each term separately.
>
> #### **Term for $ x = 1 $:**
>
> $
> P(1) \ln \left( \frac{P(1)}{Q(1)} \right) = p \ln \left( \frac{p}{q} \right)
> $
>
> #### **Term for $ x = 0 $:**
>
> $
> P(0) \ln \left( \frac{P(0)}{Q(0)} \right) = (1 - p) \ln \left( \frac{1 - p}{1 - q} \right)
> $
>
> #### **Total KL-Divergence:**
>
> $
> D_{\text{KL}}(P \parallel Q) = p \ln \left( \frac{p}{q} \right) + (1 - p) \ln \left( \frac{1 - p}{1 - q} \right)
> $
>
> ---
>
> ### **4. Applying This to Sparse Autoencoders**
>
> In the context of sparse autoencoders:
>
> - **Desired Sparsity ($ \rho $)**: This is the target average activation of the neurons. It's the "success" probability in our Bernoulli distribution $ P $.
>
> - **Actual Average Activation ($ \hat{\rho}_j $)**: This is the observed average activation for neuron $ j $ over the training data. It's the "success" probability in our Bernoulli distribution $ Q $.
>
> So, we can set:
>
> - $ p = \rho $
> - $ q = \hat{\rho}_j $
>
> #### **Plugging into the Formula:**
>
> $
> \text{KL}(\rho \,||\, \hat{\rho}_j) = \rho \ln \left( \frac{\rho}{\hat{\rho}_j} \right) + (1 - \rho) \ln \left( \frac{1 - \rho}{1 - \hat{\rho}_j} \right)
> $
>
> ---
>
> ### **Detailed Step-by-Step Derivation**
>
> Let's go through the derivation step by step with explanations.
>
> #### **Step 1: Write Down the KL-Divergence Definition for Bernoulli Distributions**
>
> $
> D_{\text{KL}}(P \parallel Q) = \sum_{x \in \{0,1\}} P(x) \ln \left( \frac{P(x)}{Q(x)} \right) = P(0) \ln \left( \frac{P(0)}{Q(0)} \right) + P(1) \ln \left( \frac{P(1)}{Q(1)} \right)
> $
>
> #### **Step 2: Substitute the PMFs**
>
> - For $ x = 0 $:
>
>   $
>   P(0) = 1 - p, \quad Q(0) = 1 - q
>   $
>
> - For $ x = 1 $:
>
>   $
>   P(1) = p, \quad Q(1) = q
>   $
>
> #### **Step 3: Plug in the Values**
>
> $
> D_{\text{KL}}(P \parallel Q) = (1 - p) \ln \left( \frac{1 - p}{1 - q} \right) + p \ln \left( \frac{p}{q} \right)
> $
>
> (Note that the order of terms is swapped compared to the earlier expression; addition is commutative.)
>
> #### **Step 4: Rewriting in Terms of $ \rho $ and $ \hat{\rho}_j $**
>
> Set $ p = \rho $ and $ q = \hat{\rho}_j $:
>
> $
> D_{\text{KL}}(\rho \,||\, \hat{\rho}_j) = (\mathbf{1} - \rho) \ln \left( \frac{\mathbf{1} - \rho}{\mathbf{1} - \hat{\rho}_j} \right) + \rho \ln \left( \frac{\rho}{\hat{\rho}_j} \right)
> $
>
> (Note: Bold numbers are to emphasize that they are constants.)
>
> #### **Step 5: Final Formula**
>
> Rewriting to match the common expression:
>
> $
> \text{KL}(\rho \,||\, \hat{\rho}_j) = \rho \ln \left( \frac{\rho}{\hat{\rho}_j} \right) + (1 - \rho) \ln \left( \frac{1 - \rho}{1 - \hat{\rho}_j} \right)
> $
>
> ---
>
> ### **Explanation of Each Component**
>
> #### **Why Use KL-Divergence Between Bernoulli Distributions?**
>
> - In sparse autoencoders, we aim to enforce that the neurons are activated only a small fraction of the time (i.e., they are sparse).
>
> - We model the activation of each neuron as a Bernoulli random variable:
>
>   - **Success (activation):** The neuron fires (activation close to 1).
>
>   - **Failure (no activation):** The neuron does not fire (activation close to 0).
>
> - By comparing the desired activation probability $ \rho $ with the actual activation probability $ \hat{\rho}_j $, we can quantify how "far" the neuron's behavior is from the desired sparsity.
>
> #### **Interpretation of the KL-Divergence Components**
>
> - **$ \rho \ln \left( \frac{\rho}{\hat{\rho}_j} \right) $:**
>
>   - This term penalizes deviations where the neuron's actual activation $ \hat{\rho}_j $ is less than the desired activation $ \rho $.
>
> - **$ (1 - \rho) \ln \left( \frac{1 - \rho}{1 - \hat{\rho}_j} \right) $:**
>
>   - This term penalizes deviations where the neuron's actual inactivation rate $ 1 - \hat{\rho}_j $ is less than the desired inactivation rate $ 1 - \rho $.
>
> - **Overall Effect:**
>
>   - The KL-divergence is zero when $ \hat{\rho}_j = \rho $.
>
>   - It increases as $ \hat{\rho}_j $ diverges from $ \rho $, penalizing the loss function accordingly.
>
> ---
>
> ### **Application in Sparse Autoencoders**
>
> #### **Loss Function with Sparsity Penalty**
>
> The total loss function $ J $ in a sparse autoencoder includes:
>
> 1. **Reconstruction Loss ($ J_{\text{recon}} $)**: Measures how well the autoencoder reconstructs the input.
>
> 2. **Weight Decay Term ($ J_{\text{weights}} $)**: Regularization term to prevent overfitting (often using L2 regularization).
>
> 3. **Sparsity Penalty ($ J_{\text{sparsity}} $)**: Sum of KL-divergence terms over all hidden neurons.
>
> $
> J = J_{\text{recon}} + \lambda J_{\text{weights}} + \beta J_{\text{sparsity}}
> $
>
> Where:
>
> - **$ \lambda $**: Weight decay parameter.
>
> - **$ \beta $**: Sparsity penalty weight.
>
> - **$ J_{\text{sparsity}} = \sum_{j=1}^{n_{\text{hidden}}} \text{KL}(\rho \,||\, \hat{\rho}_j) $**
>
> ---
>
> ### **Example Calculation**
>
> To solidify understanding, let's compute the KL-divergence for specific values.
>
> #### **Given:**
>
> - **Desired Sparsity ($ \rho $)**: 0.2
>
> - **Actual Activation ($ \hat{\rho}_j $)**: Let's say $ \hat{\rho}_j = 0.5 $
>
> #### **Compute KL-Divergence:**
>
> $
> \begin{align*}
> \text{KL}(\rho \,||\, \hat{\rho}_j) &= \rho \ln \left( \frac{\rho}{\hat{\rho}_j} \right) + (1 - \rho) \ln \left( \frac{1 - \rho}{1 - \hat{\rho}_j} \right) \\
> &= 0.2 \ln \left( \frac{0.2}{0.5} \right) + 0.8 \ln \left( \frac{0.8}{0.5} \right) \\
> &= 0.2 \ln (0.4) + 0.8 \ln (1.6) \\
> &= 0.2 \times (-0.9163) + 0.8 \times (0.4700) \\
> &= -0.1833 + 0.3760 \\
> &= 0.1927
> \end{align*}
> $
>
> - **Interpretation:**
>
>   - The KL-divergence is positive, indicating a discrepancy between the desired and actual activations.
>
>   - The loss function will include this penalty, encouraging the network to adjust $ \hat{\rho}_j $ towards $ \rho $.
>
> ---
>
> ### **Key Points to Remember**
>
> - **KL-Divergence Measures Divergence:** It's always non-negative and zero only when the two distributions are identical.
>
> - **Asymmetry:** KL-divergence is not symmetric; $ D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P) $.
>
> - **Application in Sparse Autoencoders:** The formula specifically compares the desired sparsity level with the observed average activations to enforce sparsity.
>
> ---
>
> ### **Conclusion**
>
> By deriving the KL-divergence between two Bernoulli distributions, we obtain the formula used in sparse autoencoders to penalize deviations from the desired sparsity level. This penalty helps the model learn sparse representations by adjusting the neurons' activations to match the target sparsity.
>

#### Denoising autoencoders

So far I've discussed the concept of training a neural network where the input and outputs are identical and our model is tasked with reproducing the input as closely as possible while passing through some sort of information bottleneck. Recall that I mentioned we'd like our autoencoder to be sensitive enough to recreate the original observation but insensitive enough to the training data such that the model learns a generalizable encoding and decoding. Another approach towards developing a generalizable model is to slightly corrupt the input data but still maintain the uncorrupted data as our target output.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-15 下午7.48.39.png" alt="截屏2024-09-15 下午7.48.39" style="zoom:50%;" />

With this approach, **our model isn't able to simply develop a mapping which memorizes the training data because our input and target output are no longer the same**. Rather, the model learns a vector field for mapping the input data towards a lower-dimensional manifold (recall from my earlier graphic that a manifold describes the high density region where the input data concentrates); if this manifold accurately describes the natural data, we've effectively "canceled out" the added noise.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-15 下午7.50.37.png" alt="截屏2024-09-15 下午7.50.37" style="zoom:50%;" />

The above figure visualizes the vector field described by comparing the reconstruction of $x$ with the original value of $x$. The yellow points represent training examples prior to the addition of noise. As you can see, the model has learned to adjust the corrupted input towards the learned manifold.

It's worth noting that this vector field is typically only well behaved in the regions where the model has observed during training. In areas far away from the natural data distribution, the reconstruction error is both large and does not always point in the direction of the true distribution.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-15 下午8.32.51.png" alt="截屏2024-09-15 下午8.32.51" style="zoom:50%;" />

#### Contractive autoencoders

One would expect that **for very similar inputs, the learned encoding would also be very similar**. We can explicitly train our model in order for this to be the case by requiring that the *derivative of the hidden layer activations are small* with respect to the input. In other words, for small changes to the input, we should still maintain a very similar encoded state. This is quite similar to a denoising autoencoder in the sense that these small perturbations to the input are essentially considered noise and that we would like our model to be robust against noise. Put in [other words](https://arxiv.org/abs/1211.4246?ref=jeremyjordan.me) (emphasis mine), "denoising autoencoders make the *reconstruction function* (ie. decoder) resist small but ﬁnite-sized perturbations of the input, while contractive autoencoders make the *feature extraction function* (ie. encoder) resist infinitesimal perturbations of the input."

Because we're explicitly encouraging our model to learn an encoding in which similar inputs have similar encodings, we're essentially forcing the model to learn how to **contract** a neighborhood of inputs into a smaller neighborhood of outputs. Notice how the slope (ie. derivative) of the reconstructed data is essentially zero for local neighborhoods of input data.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-15 下午8.35.19.png" alt="截屏2024-09-15 下午8.35.19" style="zoom:50%;" />

We can accomplish this by constructing a loss term which penalizes large *derivatives* of our *hidden layer activations* with respect to the input training examples, essentially penalizing instances where a small change in the input leads to a large change in the encoding space.

In fancier mathematical terms, we can craft our regularization loss term as the squared Frobenius norm ${\left\lVert A \right\rVert_F}$ of the Jacobian matrix ${\bf{J}}$ for the hidden layer activations with respect to the input observations. A Frobenius norm is essentially an L2 norm for a matrix and the Jacobian matrix simply represents all first-order partial derivatives of a vector-valued function (in this case, we have a vector of training examples).

For $m$ observations and $n$ hidden layer nodes, we can calculate these values as follows.
$$
{\left\lVert A \right\rVert_F}= \sqrt {\sum\limits_{i = 1}^m {\sum\limits_{j = 1}^n {{{\left| {{a_{ij}}} \right|}^2}} } }
$$

$$
J = \begin{bmatrix}
    \frac{\partial a_1^{(h)}(x)}{\partial x_1} & \cdots & \frac{\partial a_1^{(h)}(x)}{\partial x_m} \\
    \vdots & \ddots & \vdots \\
    \frac{\partial a_n^{(h)}(x)}{\partial x_1} & \cdots & \frac{\partial a_n^{(h)}(x)}{\partial x_m}
\end{bmatrix}
$$

Written more succinctly, we can define our complete loss function as
$$
{\cal L}\left( {x,\hat x} \right) + \lambda {\sum\limits_i {\left\lVert {{\nabla _ x}a_i^{\left( h \right)}\left( x \right)} \right\rVert} ^2}
$$
where ${{\nabla_x}a_i^{\left( h \right)}\left( x \right)}$ defines the gradient field of our hidden layer activations with respect to the input $x$ , summed over all $i$ training examples.

## Summary

An autoencoder is a neural network architecture capable of discovering structure within data in order to develop a compressed representation of the input. Many different variants of the general autoencoder architecture exist with the goal of ensuring that the compressed representation represents *meaningful* attributes of the original data input; typically the biggest challenge when working with autoencoders is getting your model to actually learn a meaningful and generalizable latent space representation.

Because autoencoders *learn* how to compress the data based on attributes (ie. correlations between the input feature vector) *discovered from data during training*, these models are typically only capable of reconstructing data similar to the class of observations of which the model observed during training.

Applications of autoencoders include:

- Anomaly detection
- Data denoising (ex. images, audio)
- Image inpainting
- Information retrieval

# Variational autoencoders 1

In my [introductory post](https://www.jeremyjordan.me/autoencoders/) on autoencoders, I discussed various models (undercomplete, sparse, denoising, contractive) which take data as input and discover some latent state representation of that data. More specifically, our input data is converted into an *encoding vector* where each dimension represents some learned attribute about the data. The most important detail to grasp here is that our encoder network is outputting a *single value* for each encoding dimension. The decoder network then subsequently takes these values and attempts to recreate the original input.

A variational autoencoder (VAE) provides a *probabilistic* manner for describing an observation in latent space. Thus, rather than building an encoder which outputs a single value to describe each latent state attribute, we'll formulate our encoder to describe a probability distribution for each latent attribute.

## Intuition

To provide an example, let's suppose we've trained an autoencoder model on a large dataset of faces with a encoding dimension of 6. An ideal autoencoder will learn descriptive attributes of faces such as skin color, whether or not the person is wearing glasses, etc. in an attempt to describe an observation in some compressed representation.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-15 下午11.50.06.png" alt="截屏2024-09-15 下午11.50.06" style="zoom:50%;" />

In the example above, we've described the input image in terms of its latent attributes using a single value to describe each attribute. However, we may prefer to represent each latent attribute as a range of possible values. For instance, what *single value* would you assign for the smile attribute if you feed in a photo of the Mona Lisa? Using a variational autoencoder, we can describe latent attributes in probabilistic terms.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-15 下午11.55.43.png" alt="截屏2024-09-15 下午11.55.43" style="zoom:50%;" />

With this approach, we'll now represent *each latent attribute* for a given input as a probability distribution. When decoding from the latent state, we'll randomly sample from each latent state distribution to generate a vector as input for our decoder model.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-15 下午11.57.38.png" alt="截屏2024-09-15 下午11.57.38" style="zoom:50%;" />

*Note: For variational autoencoders, the encoder model is sometimes referred to as the **recognition model** whereas the decoder model is sometimes referred to as the **generative model**.*

By constructing our encoder model to output a range of possible values (a statistical distribution) from which we'll randomly sample to feed into our decoder model, we're essentially enforcing a continuous, smooth latent space representation. For any sampling of the latent distributions, we're expecting our decoder model to be able to accurately reconstruct the input. Thus, values which are nearby to one another in latent space should correspond with very similar reconstructions.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 上午12.34.51.png" alt="截屏2024-09-16 上午12.34.51" style="zoom:50%;" />

## Statisical motivation

Suppose that there exists some hidden variable $z$ which generates an observation $x$.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 上午12.36.04.png" alt="截屏2024-09-16 上午12.36.04" style="zoom:50%;" />

We can only see $x$, but we would like to infer the characteristics of $z$. In other words, we’d like to compute $p\left( {z|x} \right)$.
$$
p\left( {z|x} \right) = \frac{{p\left( {x|z} \right)p\left( z \right)}}{{p\left( x \right)}}
$$
Unfortunately, computing $p\left( x \right)$ is quite difficult.
$$
p\left( x \right) = \int {p\left( {x|z} \right)p\left( z \right)dz}
$$
This usually turns out to be an [intractable distribution](https://stats.stackexchange.com/questions/4417/intractable-posterior-distributions?ref=jeremyjordan.me). However, we can apply [varitational inference](https://arxiv.org/pdf/1601.00670.pdf?ref=jeremyjordan.me) to estimate this value.

Let's approximate $p\left( {z|x} \right)$ by another distribution $q\left( {z|x} \right)$ which we'll define such that it has a tractable distribution. If we can define the parameters of $q\left( {z|x} \right)$ such that it is very similar to $p\left( {z|x} \right)$, we can use it to perform approximate inference of the intractable distribution.

Recall that the KL divergence is a measure of difference between two probability distributions. Thus, if we wanted to ensure that $q\left( {z|x} \right)$ was similar to $p\left( {z|x} \right)$, we could minimize the KL divergence between the two distributions.
$$
\min KL\left( {q\left( {z|x} \right)||p\left( {z|x} \right)} \right)
$$
Dr. Ali Ghodsi goes through a full derivation [here](https://youtu.be/uaaqyVS9-rM?t=19m42s&ref=jeremyjordan.me), but the result gives us that we can minimize the above expression by maximizing the following:
$$
{E_{q\left( {z|x} \right)}}\log p\left( {x|z} \right) - KL\left( {q\left( {z|x} \right)||p\left( z \right)} \right)
$$

The first term represents the reconstruction likelihood and the second term ensures that our learned distribution $q$ is similar to the true prior distribution $p$.

> **Intuition of this loss function**
>
> ---
>
> ### **Understanding the VAE Loss Function**
>
> The loss function you mentioned for the VAE is:
>
> $
> \mathcal{L}(\theta, \phi; x) = \underbrace{\mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) \right]}_{\text{Reconstruction Loss}} - \underbrace{D_{\text{KL}} \left( q_\phi(z|x) \,||\, p(z) \right)}_{\text{Regularization Term}}
> $
>
> **Where:**
>
> - **$ x $**: Input data.
> - **$ z $**: Latent variables.
> - **$ \theta $** and **$ \phi $**: Parameters of the neural networks (decoder and encoder, respectively).
> - **$ q_\phi(z|x) $**: Approximate posterior distribution (encoder), parameterized by $ \phi $.
> - **$ p_\theta(x|z) $**: Likelihood of data given latent variables (decoder), parameterized by $ \theta $.
> - **$ p(z) $**: Prior distribution over latent variables (usually standard normal distribution).
> - **$ \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) \right] $**: Expected log-likelihood of reconstructing $ x $ given $ z $.
> - **$ D_{\text{KL}} \left( q_\phi(z|x) \,||\, p(z) \right) $**: Kullback-Leibler divergence between the approximate posterior and the prior.
>
> ---
>
> ### **Intuition Behind the VAE Loss Function**
>
> The VAE loss function is derived from the **Evidence Lower BOund (ELBO)** on the marginal likelihood $ \log p_\theta(x) $:
>
> $
> \log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) \right] - D_{\text{KL}} \left( q_\phi(z|x) \,||\, p(z) \right)
> $
>
> **Objective:** Maximize the ELBO with respect to $ \theta $ and $ \phi $, which simultaneously maximizes the likelihood of the data and ensures the approximate posterior $ q_\phi(z|x) $ is close to the prior $ p(z) $.
>
> **Components:**
>
> 1. **Reconstruction Loss ($ \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) \right] $):**
>
>    - Measures how well the decoder can reconstruct the input data $ x $ from the latent representation $ z $.
>    - Encourages the model to learn meaningful latent representations that capture the underlying structure of the data.
>
> 2. **Regularization Term ($ D_{\text{KL}} \left( q_\phi(z|x) \,||\, p(z) \right) $):**
>
>    - Measures how much the approximate posterior deviates from the prior distribution.
>    - Encourages the latent representations $ z $ to follow the prior distribution $ p(z) $ (usually a standard normal distribution).
>    - Promotes smoothness and continuity in the latent space, enabling the generation of new data samples by sampling $ z $ from $ p(z) $.
>
> ---
>
> ### **Step-by-Step Explanation**
>
> #### **1. The Encoder ($ q_\phi(z|x) $)**
>
> - Learns to approximate the true posterior distribution $ p(z|x) $.
> - Outputs parameters of the distribution (e.g., mean and variance if Gaussian).
>
> #### **2. The Decoder ($ p_\theta(x|z) $)**
>
> - Learns to reconstruct the input data $ x $ from latent variables $ z $.
> - Defines a likelihood function of $ x $ given $ z $.
>
> #### **3. The KL-Divergence Term**
>
> - Ensures that the learned latent representations $ z $ do not deviate too much from the prior $ p(z) $.
> - Acts as a regularizer in the loss function.
>
> ---
>
> ### **Example to Illustrate the VAE Loss Function**
>
> Let's create a simple example with mock data to understand how each term of the VAE loss function works.
>
> #### **Setup:**
>
> - **Data $ x $:** Let's assume $ x $ is a one-dimensional continuous variable.
> - **Latent Variable $ z $:** Also one-dimensional.
> - **Prior $ p(z) $:** Standard normal distribution $ \mathcal{N}(0, 1) $.
> - **Encoder $ q_\phi(z|x) $:** Outputs a Gaussian distribution $ \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x)) $.
> - **Decoder $ p_\theta(x|z) $:** Outputs a Gaussian distribution $ \mathcal{N}(\mu_\theta(z), \sigma_\theta^2) $.
>
> For simplicity, let's fix $ \sigma_\theta^2 = 1 $ and $ \sigma_\phi^2(x) = \sigma_\phi^2 $ (i.e., constant variance).
>
> #### **Step 1: Define Mock Data**
>
> Suppose we have a single data point:
>
> - **$ x = 2.0 $**
>
> #### **Step 2: Encoder Forward Pass**
>
> The encoder processes $ x $ and outputs parameters for $ q_\phi(z|x) $:
>
> - **$ \mu_\phi(x) $:** Mean of $ z $ given $ x $. Let's say $ \mu_\phi(x) = 1.5 $.
> - **$ \sigma_\phi^2 $:** Variance of $ z $ given $ x $. Let's set $ \sigma_\phi^2 = 0.25 $.
>
> So, $ q_\phi(z|x) = \mathcal{N}(z; 1.5, 0.25) $.
>
> #### **Step 3: Sampling $ z $ from $ q_\phi(z|x) $**
>
> To compute the reconstruction loss, we need to sample $ z $ from $ q_\phi(z|x) $.
>
> - Let's sample $ z = 1.6 $ from $ \mathcal{N}(1.5, 0.25) $.
>
> #### **Step 4: Decoder Forward Pass**
>
> The decoder takes $ z $ and outputs parameters for $ p_\theta(x|z) $:
>
> - **$ \mu_\theta(z) $:** Mean of $ x $ given $ z $. Let's define $ \mu_\theta(z) = W z + b $, where $ W $ and $ b $ are decoder parameters.
>
> Assume:
>
> - **$ W = 1.0 $**
> - **$ b = 0.0 $**
>   
>
> So, $ \mu_\theta(z) = 1.0 \times 1.6 + 0.0 = 1.6 $.
>
> Thus, $ p_\theta(x|z) = \mathcal{N}(x; 1.6, 1) $.
>
> #### **Step 5: Compute Reconstruction Loss**
>
> The reconstruction loss is the negative expected log-likelihood:
>
> $
> \text{Reconstruction Loss} = -\mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) \right]
> $
>
> Since we have a Gaussian likelihood, the log-likelihood is:
>
> $
> \log p_\theta(x|z) = -\frac{1}{2} \ln(2\pi \sigma_\theta^2) - \frac{(x - \mu_\theta(z))^2}{2\sigma_\theta^2}
> $
>
> Plugging in the values:
>
> - $ x = 2.0 $
> - $ \mu_\theta(z) = 1.6 $
> - $ \sigma_\theta^2 = 1 $
>
> Compute the log-likelihood:
>
> $
> \begin{align*}
> \log p_\theta(x|z) &= -\frac{1}{2} \ln(2\pi \times 1) - \frac{(2.0 - 1.6)^2}{2 \times 1} \\
> &= -\frac{1}{2} \ln(2\pi) - \frac{(0.4)^2}{2} \\
> &= -0.9189 - \frac{0.16}{2} \\
> &= -0.9189 - 0.08 \\
> &= -0.9989
> \end{align*}
> $
>
> So, the reconstruction loss is:
>
> $
> \text{Reconstruction Loss} = -(-0.9989) = 0.9989
> $
>
> #### **Step 6: Compute KL-Divergence**
>
> For Gaussian distributions, the KL-divergence between $ q_\phi(z|x) = \mathcal{N}(\mu_\phi, \sigma_\phi^2) $ and $ p(z) = \mathcal{N}(0, 1) $ is:
>
> $
> D_{\text{KL}} \left( q_\phi(z|x) \,||\, p(z) \right) = \frac{1}{2} \left( \frac{\sigma_\phi^2 + \mu_\phi^2}{1} - 1 - \ln \sigma_\phi^2 \right)
> $
>
> Plugging in the values:
>
> - $ \mu_\phi = 1.5 $
> - $ \sigma_\phi^2 = 0.25 $
>
> Compute the KL-divergence:
>
> $
> \begin{align*}
> D_{\text{KL}} &= \frac{1}{2} \left( \frac{0.25 + (1.5)^2}{1} - 1 - \ln 0.25 \right) \\
> &= \frac{1}{2} \left( 0.25 + 2.25 - 1 - \ln 0.25 \right) \\
> &= \frac{1}{2} \left( 1.5 - \ln 0.25 \right) \\
> &= \frac{1}{2} \left( 1.5 - (-1.3863) \right) \\
> &= \frac{1}{2} \left( 1.5 + 1.3863 \right) \\
> &= \frac{1}{2} \times 2.8863 \\
> &= 1.4431
> \end{align*}
> $
>
> #### **Step 7: Compute Total Loss**
>
> The total loss is:
>
> $
> \mathcal{L}(\theta, \phi; x) = \text{Reconstruction Loss} + \text{KL-Divergence}
> $
>
> Plugging in the values:
>
> $
> \mathcal{L}(\theta, \phi; x) = 0.9989 + 1.4431 = 2.4420
> $
>
> ---
>
> ### **Interpretation of the Example**
>
> - **Reconstruction Loss (0.9989):** Measures how well the decoder reconstructs $ x = 2.0 $ from the sampled $ z = 1.6 $. The closer $ \mu_\theta(z) $ is to $ x $, the lower this loss.
>
> - **KL-Divergence (1.4431):** Measures how much the approximate posterior $ q_\phi(z|x) = \mathcal{N}(1.5, 0.25) $ deviates from the prior $ p(z) = \mathcal{N}(0, 1) $. A higher KL-divergence indicates that $ q_\phi(z|x) $ is far from $ p(z) $, and the model will be penalized to bring $ q_\phi(z|x) $ closer to $ p(z) $.
>
> - **Total Loss (2.4420):** The sum of the reconstruction loss and the KL-divergence, which the VAE aims to minimize during training.
>
> ---
>
> ### **Why Does This Loss Function Make Sense?**
>
> - **Balancing Reconstruction and Regularization:**
>
>   - The reconstruction loss ensures that the model can accurately reconstruct the input data from the latent variables.
>   - The KL-divergence acts as a regularizer, pushing the approximate posterior towards the prior. This prevents overfitting and encourages the latent space to have desirable properties (e.g., continuity, smoothness).
>
> - **Generative Capability:**
>
>   - By encouraging $ q_\phi(z|x) $ to be close to $ p(z) $, we can sample $ z $ from $ p(z) $ during generation and have confidence that the decoder $ p_\theta(x|z) $ can produce realistic data samples.
>
> ---
>
> ### **Further Clarifications**
>
> #### **Why Use KL-Divergence?**
>
> - KL-divergence measures how one probability distribution diverges from another.
> - In VAEs, we use it to measure the difference between the approximate posterior $ q_\phi(z|x) $ and the prior $ p(z) $.
> - Minimizing the KL-divergence ensures that $ q_\phi(z|x) $ remains close to $ p(z) $, which is essential for generating new data samples.
>
> #### **Reparameterization Trick**
>
> - In practice, to compute gradients and backpropagate through stochastic nodes, we use the reparameterization trick.
> - Instead of sampling $ z \sim q_\phi(z|x) $ directly, we sample $ \epsilon \sim \mathcal{N}(0, 1) $ and compute $ z = \mu_\phi(x) + \sigma_\phi(x) \epsilon $.
> - This allows gradients to flow through $ \mu_\phi(x) $ and $ \sigma_\phi(x) $ during training.
>
> ---
>
> ### **Summary**
>
> - **VAE Loss Function:**
>
>   $
>   \mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)} \left[ \log p_\theta(x|z) \right] - D_{\text{KL}} \left( q_\phi(z|x) \,||\, p(z) \right)
>   $
>
> - **Objective:**
>
>   - Maximize the likelihood of the data while keeping the approximate posterior close to the prior.
>   - Achieved by minimizing the total loss (negative ELBO).
>
> - **Components:**
>
>   - **Reconstruction Loss:** Encourages accurate reconstruction of input data.
>   - **KL-Divergence:** Regularizes the latent space by aligning $ q_\phi(z|x) $ with $ p(z) $.
>
> - **Example Walkthrough:**
>
>   - Demonstrated how to compute each term with mock data.
>   - Illustrated the interplay between reconstruction accuracy and regularization.
>

****

To revisit our graphical model, we can use $q$ to infer the possible hidden variables (ie. latent state) which was used to generate an observation. We can further construct this model into a neural network architecture where the encoder model learns a mapping from $x$ to $z$ and the decoder model learns a mapping from $z$ back to $x$.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午8.50.25.png" alt="截屏2024-09-16 下午8.50.25" style="zoom:50%;" />

Our loss function for this network will consist of two terms, one which penalizes reconstruction error (which can be thought of maximizing the reconstruction likelihood as discussed earlier) and a second term which encourages our learned distribution ${q\left( {z|x} \right)}$ to be similar to the true prior distribution ${p\left( z \right)}$, which we'll assume follows a unit Gaussian distribution, for each dimension $j$ of the latent space.
$$
{\cal L}\left( {x,\hat x} \right) + \sum\limits_j {KL\left( {{q_j}\left( {z|x} \right)||p\left( z \right)} \right)}
$$

## Implementation

In the previous section, I established the statistical motivation for a variational autoencoder structure. In this section, I'll provide the practical implementation details for building such a model yourself.

Rather than directly outputting values for the latent state as we would in a standard autoencoder, the encoder model of a VAE will output parameters describing a distribution for each dimension in the latent space. Since we're assuming that our prior follows a normal distribution, we'll output *two* vectors describing the mean and variance of the latent state distributions. If we were to build a true multivariate Gaussian model, we'd need to define a covariance matrix describing how each of the dimensions are correlated. However, we'll make a simplifying assumption that our covariance matrix only has nonzero values on the diagonal, allowing us to describe this information in a simple vector.

Our decoder model will then generate a latent vector by sampling from these defined distributions and proceed to develop a reconstruction of the original input.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午8.56.15.png" alt="截屏2024-09-16 下午8.56.15" style="zoom:50%;" />

The sample process can be Zoom-in as following:

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午9.27.45.png" alt="截屏2024-09-16 下午9.27.45" style="zoom:50%;" />

This stochastic generation means, that even for the same input, while the mean and standard deviations remain the same, the actual encoding will somewhat vary on every single pass simply due to sampling.

However, this sampling process requires some extra attention. When training the model, we need to be able to calculate the relationship of each parameter in the network with respect to the final output loss using a technique known as [backpropagation](https://www.jeremyjordan.me/neural-networks-training/). However, we simply cannot do this for a *random sampling* process. Fortunately, we can leverage a clever idea known as the "reparameterization trick" which suggests that we randomly sample $\epsilon$ from a unit Gaussian, and then shift the randomly sampled $\epsilon$ by the latent distribution's mean $\mu$ and scale it by the latent distribution's variance $\sigma$.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午9.11.56.png" alt="截屏2024-09-16 下午9.11.56" style="zoom:50%;" />

With this reparameterization, we can now optimize the *parameters* of the distribution while still maintaining the ability to randomly sample from that distribution.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午9.12.22.png" alt="截屏2024-09-16 下午9.12.22" style="zoom:50%;" />

*Note: In order to deal with the fact that the network may learn negative values for $\sigma$, we'll typically have the network learn $\log \sigma$ and exponentiate this value to get the latent distribution's variance.* 

## Visualization of latent space

To understand the implications of a variational autoencoder model and how it differs from standard autoencoder architectures, it's useful to examine the latent space. [This blog post](https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf?ref=jeremyjordan.me) introduces a great discussion on the topic, which I'll summarize in this section.

The main benefit of a variational autoencoder is that we're capable of learning *smooth* latent state representations of the input data. For standard autoencoders, we simply need to learn an encoding which allows us to reproduce the input. As you can see in the left-most figure, focusing only on reconstruction loss *does* allow us to separate out the classes (in this case, MNIST digits) which should allow our decoder model the ability to reproduce the original handwritten digit, but there's an uneven distribution of data within the latent space. In other words, there are areas in latent space which don't represent *any* of our observed data.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午9.25.32.png" alt="截屏2024-09-16 下午9.25.32" style="zoom:50%;" />

For example, training an autoencoder on the MNIST dataset, and visualizing the encodings from a 2D latent space reveals the formation of distinct clusters. This makes sense, as distinct encodings for each image type makes it far easier for the decoder to decode them. This is fine if you’re just *replicating* the same images.

But when you’re building a *generative* model, you **don’t** want to prepare to *replicate* the same image you put in. You want to randomly sample from the latent space, or generate variations on an input image, from a continuous latent space.

If the space has discontinuities (eg. gaps between clusters) and you sample/generate a variation from there, the decoder will simply generate an unrealistic output, because the decoder has *no idea* how to deal with that region of the latent space. During training, it *never saw* encoded vectors coming from that region of latent space.

On the flip side, if we only focus only on ensuring that the latent distribution is similar to the prior distribution (through our KL divergence loss term), we end up describing *every* observation using the same unit Gaussian, which we subsequently sample from to describe the latent dimensions visualized. This effectively treats every observation as having the same characteristics; in other words, we've failed to describe the original data.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午9.43.42.png" alt="截屏2024-09-16 下午9.43.42" style="zoom:50%;" />

However, when the two terms are optimized simultaneously, we're encouraged to describe the latent state for an observation with distributions close to the prior but deviating when necessary to describe salient features of the input.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午9.46.17.png" alt="截屏2024-09-16 下午9.46.17" style="zoom:50%;" />

Intuitively, this is the equilibrium reached by the *cluster-forming* nature of the reconstruction loss, and the *dense packing* nature of the KL loss, forming distinct clusters the decoder can decode. This is great, as it means when randomly generating, if you sample a vector from the same prior distribution of the encoded vectors, *N*(**0**, **I**), the decoder will successfully decode it. And if you’re interpolating, there are no sudden gaps between clusters, but a *smooth mix of features* a decoder can understand.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午9.48.41.png" alt="截屏2024-09-16 下午9.48.41" style="zoom:50%;" />

When I'm constructing a variational autoencoder, I like to inspect the latent dimensions for a few samples from the data to see the characteristics of the distribution. I encourage you to do the same.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-16 下午9.49.38.png" alt="截屏2024-09-16 下午9.49.38" style="zoom:50%;" />

If we observe that the latent distributions appear to be very tight, we may decide to give higher weight to the KL divergence term with a parameter $\beta>1$, encouraging the network to learn broader distributions. This simple insight has led to the growth of a new class of models - disentangled variational autoencoders. As it turns out, by placing a larger emphasis on the KL divergence term we're also implicitly enforcing that the learned latent dimensions are uncorrelated (through our simplifying assumption of a diagonal covariance matrix).
$$
{\cal L}\left( {x,\hat x} \right) + \beta \sum\limits_j {KL\left( {{q_j}\left( {z|x} \right)||N\left( {0,1} \right)} \right)}
$$

## Variational autoencoders as a generative model

By sampling from the latent space, we can use the decoder network to form a generative model capable of creating new data similar to what was observed during training. Specifically, we'll sample from the prior distribution ${p\left( z \right)}$ which we assumed follows a unit Gaussian distribution.

The figure below visualizes the data generated by the decoder network of a variational autoencoder trained on the MNIST handwritten digits dataset. Here, we've sampled a grid of values from a two-dimensional Gaussian and displayed the output of our decoder network.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-17 上午12.16.49.png" alt="截屏2024-09-17 上午12.16.49" style="zoom:50%;" />

As you can see, the distinct digits each exist in different regions of the latent space and smoothly transform from one digit to another. This smooth transformation can be quite useful when you'd like to interpolate between two observations, such as this recent example where [Google built a model for interpolating between two music samples](https://magenta.tensorflow.org/music-vae?ref=jeremyjordan.me).

# VAE: Variational Autoencoder 2

The idea of **Variational Autoencoder** ([Kingma & Welling, 2014](https://arxiv.org/abs/1312.6114)), short for **VAE**, is actually less similar to all the autoencoder models above, but deeply rooted in the methods of variational bayesian and graphical model.

Instead of mapping the input into a *fixed* vector, we want to map it into a distribution. Let’s label this distribution as $p_\theta$, parameterized by $\theta$. The relationship between the data input $\mathbf{x}$ and the latent encoding vector $\mathbf{z}$ can be fully defined by:

- Prior $p_\theta(\mathbf{z})$
- Likelihood $p_\theta(\mathbf{x}\vert\mathbf{z})$
- Posterior $p_\theta(\mathbf{z}\vert\mathbf{x})$

Assuming that we know the real parameter $\theta^{*}$ for this distribution. In order to generate a sample that looks like a real data point $\mathbf{x}^{(i)}$, we follow these steps:

1. First, sample a $\mathbf{z}^{(i)}$ from a prior distribution $p_{\theta^*}(\mathbf{z})$. 
   - $\mathbf{z}^{(i)} \sim p_\theta(\mathbf{z})$ 
   - $ p_\theta(\mathbf{z}) $ is the **prior distribution** over the latent variables $ \mathbf{z} $, often chosen to be a standard normal distribution $ \mathcal{N}(0, \mathbf{I}) $.
2. Then a value is generated from a conditional distribution $p_{\theta^*}(\mathbf{x} \vert \mathbf{z} = \mathbf{z}^{(i)})$.
   - $\mathbf{x}^{(i)} \sim p_\theta(\mathbf{x} | \mathbf{z}^{(i)})$
   - $ p_\theta(\mathbf{x} | \mathbf{z}) $ is the **likelihood function** or **decoder** that models the distribution of observations given the latent variables.

The goal during training is to find the parameter $ \theta^{*} $ that maximizes the likelihood of the observed data under the model. The optimal parameter $\theta^{*}$ is the one that maximizes the probability of generating real data samples (**Maximum Likelihood Estimation (MLE)**):
$$
\theta^{*} = \arg\max_\theta \prod_{i=1}^n p_\theta(\mathbf{x}^{(i)})
$$
Commonly we use the log probabilities to convert the product on RHS to a sum (Check the Note ML4 for why using log):
$$
\theta^{*} = \arg\max_\theta \sum_{i=1}^n \log p_\theta(\mathbf{x}^{(i)})
$$
Now let’s update the equation to better demonstrate the data generation process so as to involve the encoding vector:

The goal during training is to find the parameter $ \theta^{*} $ that maximizes the likelihood of the observed data under the model. However, directly computing $ p_\theta(\mathbf{x}^{(i)}) $ involves integrating over all possible latent variables $ \mathbf{z} $:
$$
p_\theta(\mathbf{x}^{(i)}) = \int p_\theta(\mathbf{x}^{(i)}\vert\mathbf{z}) p_\theta(\mathbf{z}) d\mathbf{z}
$$
Unfortunately it is not easy to compute $p_\theta(\mathbf{x}^{(i)})$ in this way, as it is very expensive to check all the possible values of $\mathbf{z}$ and sum them up. This integral is often **intractable** due to the complexity of the model and the high dimensionality of $ \mathbf{z}$. To narrow down the value space to facilitate faster search, we would like to introduce a new approximation function to output what is a likely code given an input $\mathbf{x}$, $q_\phi(\mathbf{z}\vert\mathbf{x})$, parameterized by $\phi$.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-17 下午6.18.56.png" alt="截屏2024-09-17 下午6.18.56" style="zoom:50%;" />

Now the structure looks a lot like an autoencoder:

- The conditional probability $p_\theta(\mathbf{x} \vert \mathbf{z})$ defines a generative model, similar to the decoder $f_\theta(\mathbf{x} \vert \mathbf{z})$ introduced above. $p_\theta(\mathbf{x} \vert \mathbf{z})$ is also known as *probabilistic decoder*.
- The approximation function $q_\phi(\mathbf{z} \vert \mathbf{x})$ is the *probabilistic encoder*, playing a similar role as $g_\phi(\mathbf{z} \vert \mathbf{x})$ above.

## Loss Function: ELBO[#](https://lilianweng.github.io/posts/2018-08-12-vae/#loss-function-elbo)

The estimated posterior $q_\phi(\mathbf{z}\vert\mathbf{x})$ should be very close to the real one $p_\theta(\mathbf{z}\vert\mathbf{x})$. We can use [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) to quantify the distance between these two distributions. KL divergence $D_\text{KL}(X|Y)$ measures how much information is lost if the distribution Y is used to represent X.

In our case we want to minimize $D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) | p_\theta(\mathbf{z}\vert\mathbf{x}) )$ with respect to $\phi$.

But why use $D_\text{KL}(q_\phi | p_\theta)$(reversed KL) instead of $D_\text{KL}(p_\theta | q_\phi)$(forward KL)? Eric Jang has a great explanation in his [post](https://blog.evjang.com/2016/08/variational-bayes.html) on Bayesian Variational methods (Check the next section).

Let’s now expand the equation:
$$
\begin{aligned}
& D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z} \vert \mathbf{x})} d\mathbf{z} & \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})p_\theta(\mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z \vert x) = p(z, x) / p(x)} \\
&=\int q_\phi(\mathbf{z} \vert \mathbf{x}) \big( \log p_\theta(\mathbf{x}) + \log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \big) d\mathbf{z} & \\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} d\mathbf{z} & \scriptstyle{\text{; Because }\int q(z \vert x) dz = 1 (*)}\\
&=\log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x})\log\frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{x}\vert\mathbf{z})p_\theta(\mathbf{z})} d\mathbf{z} & \scriptstyle{\text{; Because }p(z, x) = p(x \vert z) p(z)} \\
&=\log p_\theta(\mathbf{x}) + \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z} \vert \mathbf{x})}[\log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z})} - \log p_\theta(\mathbf{x} \vert \mathbf{z})] &\\
&=\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) &
\end{aligned}
$$

> # Some interpretation about * step
>
> ### **Step-by-Step Derivation**
>
> #### **Understanding the Integral**
>
> First, let's carefully analyze the first expression:
>
> $$
> I = \int q_\phi(\mathbf{z} \vert \mathbf{x}) \left( \log p_\theta(\mathbf{x}) + \log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \right) d\mathbf{z}
> $$
> Our goal is to simplify this integral and see how it leads to the second expression.
>
> #### **Step 1: Recognize Constants with Respect to $ \mathbf{z} $**
>
> Notice that:
>
> - **$ \log p_\theta(\mathbf{x}) $** is a constant with respect to $ \mathbf{z} $ because it depends only on $ \mathbf{x} $ and $ \theta $.
>   - **Explanation**: Since $ p_\theta(\mathbf{x}) $ is the marginal likelihood of the observed data $ \mathbf{x} $, it's not a function of $ \mathbf{z} $.
>
> #### **Step 2: Use Linearity of Integration**
>
> We can use the linearity property of integrals, which states that:
>
> $$
> \int f(\mathbf{z}) + g(\mathbf{z}) \, d\mathbf{z} = \int f(\mathbf{z}) \, d\mathbf{z} + \int g(\mathbf{z}) \, d\mathbf{z}
> $$
> So, we can split the integral into two parts:
>
> $$
> I = \int q_\phi(\mathbf{z} \vert \mathbf{x}) \log p_\theta(\mathbf{x}) \, d\mathbf{z} + \int q_\phi(\mathbf{z} \vert \mathbf{x}) \log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \, d\mathbf{z}
> $$
>
> #### **Step 3: Simplify the First Integral**
>
> Since $ \log p_\theta(\mathbf{x}) $ is a constant with respect to $ \mathbf{z} $, we can factor it out of the integral:
>
> $$
> \int q_\phi(\mathbf{z} \vert \mathbf{x}) \log p_\theta(\mathbf{x}) \, d\mathbf{z} = \log p_\theta(\mathbf{x}) \int q_\phi(\mathbf{z} \vert \mathbf{x}) \, d\mathbf{z}
> $$
>
> #### **Step 4: Recognize the Integral of a Probability Density**
>
> The integral of a probability density function (PDF) over its entire support equals 1:
>
> $$
> \int q_\phi(\mathbf{z} \vert \mathbf{x}) \, d\mathbf{z} = 1
> $$
> **Explanation**:
>
> - $ q_\phi(\mathbf{z} \vert \mathbf{x}) $ is a valid probability distribution over $ \mathbf{z} $ given $ \mathbf{x} $.
> - The total probability over all possible values of $ \mathbf{z} $ must sum (or integrate) to 1.
>
> #### **Step 5: Simplify the First Integral Using This Property**
>
> Therefore, the first integral simplifies to:
>
> $$
> \log p_\theta(\mathbf{x}) \int q_\phi(\mathbf{z} \vert \mathbf{x}) \, d\mathbf{z} = \log p_\theta(\mathbf{x}) \times 1 = \log p_\theta(\mathbf{x})
> $$
>
> #### **Step 6: Write the Simplified Expression**
>
> Now, the entire expression simplifies to:
>
> $$
> I = \log p_\theta(\mathbf{x}) + \int q_\phi(\mathbf{z} \vert \mathbf{x}) \log \frac{q_\phi(\mathbf{z} \vert \mathbf{x})}{p_\theta(\mathbf{z}, \mathbf{x})} \, d\mathbf{z}
> $$

So we have:
$$
D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) =\log p_\theta(\mathbf{x}) + D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z})) - \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z})
$$
Once rearrange the left and right hand side of the equation,
$$
\log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) = \mathbb{E}_{\mathbf{z}\sim q_\phi(\mathbf{z}\vert\mathbf{x})}\log p_\theta(\mathbf{x}\vert\mathbf{z}) - D_\text{KL}(q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}))
$$

The LHS of the equation is exactly what we want to maximize when learning the true distributions: we want to maximize the (log-)likelihood of generating real data (that is $\log p_\theta(\mathbf{x})$) and also minimize the difference between the real and estimated posterior distributions (the term $D_\text{KL}$ works like a regularizer). Note that $p_\theta(\mathbf{x})$ is fixed with respect to $q_\phi$.

 The negation of the above defines our loss function:
$$
\begin{aligned}
L_\text{VAE}(\theta, \phi) 
&= -\log p_\theta(\mathbf{x}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) )\\
&= - \mathbb{E}_{\mathbf{z} \sim q_\phi(\mathbf{z}\vert\mathbf{x})} \log p_\theta(\mathbf{x}\vert\mathbf{z}) + D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}) ) \\
\theta^{*}, \phi^{*} &= \arg\min_{\theta, \phi} L_\text{VAE}
\end{aligned}
$$
In Variational Bayesian methods, this loss function is known as the *variational lower bound*, or *evidence lower bound*. The “lower bound” part in the name comes from the fact that KL divergence is always non-negative and thus $-L_\text{VAE}$ is the lower bound of $\log p_\theta (\mathbf{x})$.
$$
-L_\text{VAE} = \log p_\theta(\mathbf{x}) - D_\text{KL}( q_\phi(\mathbf{z}\vert\mathbf{x}) \| p_\theta(\mathbf{z}\vert\mathbf{x}) ) \leq \log p_\theta(\mathbf{x})
$$
Therefore by minimizing the loss, we are maximizing the lower bound of the probability of generating real data samples.

# Actual loss function implemented 

## Summary of Your Question

You have derived the VAE loss function as:

$$
\begin{aligned}
L_\text{VAE}(\theta, \phi) 
&= -\log p_\theta(\mathbf{x}) + D_\text{KL}( q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x}) )\\
&= - \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right] + D_\text{KL}( q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}) )
\end{aligned}
$$

However, in practical implementations, the loss function is simplified to:

```python
def loss_function(x_reconstructed, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    BCE = nn.functional.binary_cross_entropy(
        x_reconstructed, x, reduction='sum')

    # KL divergence between the learned latent distribution and standard normal distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return BCE + KLD
```

**Your Question:** Why do we use this simplified loss function in practice instead of the full theoretical expression?

---

## **Answer**

The simplified loss function used in practical VAE implementations is derived from the theoretical loss function you mentioned. The simplification arises due to specific choices of probability distributions and assumptions made in the VAE model. Below, I'll explain how the theoretical loss function simplifies to the practical implementation, step by step.

---

### **1. Understanding the Theoretical Loss Function**

The theoretical VAE loss function consists of two main components:

1. **Reconstruction Loss:**
   $$
   - \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right]
   $$

2. **Regularization Loss (KL Divergence):**
   $$
   D_\text{KL}( q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}) )
   $$

Our goal is to:

- **Maximize** the expected log-likelihood of the data given the latent variables (i.e., minimize the reconstruction loss).
- **Minimize** the KL divergence between the approximate posterior $ q_\phi(\mathbf{z}|\mathbf{x}) $ and the prior $ p(\mathbf{z}) $.

---

### **2. Assumptions in Practical Implementations**

To make the loss function tractable and implementable, we make specific assumptions about the distributions involved:

#### **a. Prior Distribution $ p(\mathbf{z}) $**

- **Assumption:** The prior over the latent variables $ \mathbf{z} $ is a standard multivariate normal distribution:
  $$
  p(\mathbf{z}) = \mathcal{N}(\mathbf{z}; \mathbf{0}, \mathbf{I})
  $$
- **Reason:** A standard normal prior simplifies the computation of the KL divergence and encourages the latent space to be well-behaved.

#### **b. Approximate Posterior $ q_\phi(\mathbf{z}|\mathbf{x}) $**

- **Assumption:** The approximate posterior is a multivariate normal distribution with a diagonal covariance matrix:
  $$
  q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \operatorname{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))
  $$
  - $ \boldsymbol{\mu}_\phi(\mathbf{x}) $: Mean vector output by the encoder network.
  - $ \boldsymbol{\sigma}^2_\phi(\mathbf{x}) $: Variance vector (diagonal elements of the covariance matrix) output by the encoder network.

#### **c. Likelihood Model $ p_\theta(\mathbf{x}|\mathbf{z}) $**

- **Assumption:** The likelihood of the data given the latent variables is modeled using a specific distribution, often chosen based on the data type.
  - For **binary data** (e.g., black-and-white images), we use a Bernoulli distribution, leading to a binary cross-entropy (BCE) loss.
  - For **continuous data**, we might use a Gaussian distribution, leading to a mean squared error (MSE) loss.

---

### **3. Simplifying the KL Divergence Term**

Under the assumptions above, the KL divergence between two multivariate normal distributions with diagonal covariance matrices can be computed analytically.

#### **a. KL Divergence Between Two Gaussians**

For two multivariate normal distributions $ q = \mathcal{N}(\boldsymbol{\mu}_q, \operatorname{diag}(\boldsymbol{\sigma}^2_q)) $ and $ p = \mathcal{N}(\boldsymbol{\mu}_p, \operatorname{diag}(\boldsymbol{\sigma}^2_p)) $, the KL divergence is:

$$
D_\text{KL}(q \| p) = \frac{1}{2} \sum_{i=1}^{D} \left( \log \frac{\sigma^2_{p,i}}{\sigma^2_{q,i}} - 1 + \frac{\sigma^2_{q,i}}{\sigma^2_{p,i}} + \frac{(\mu_{q,i} - \mu_{p,i})^2}{\sigma^2_{p,i}} \right)
$$

#### **b. Simplifying with Standard Normal Prior**

Since $ p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I}) $, we have $ \mu_{p,i} = 0 $ and $ \sigma^2_{p,i} = 1 $ for all $ i $.

Plugging in these values:

$$
D_\text{KL}( q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}) ) = \frac{1}{2} \sum_{i=1}^{D} \left( - \log \sigma^2_{q,i} - 1 + \sigma^2_{q,i} + \mu_{q,i}^2 \right)
$$

#### **c. Implementing in Code**

In the code, we have:

- $ \text{mu} = \boldsymbol{\mu}_\phi(\mathbf{x}) $
- $ \text{logvar} = \log \boldsymbol{\sigma}^2_\phi(\mathbf{x}) $

The KL divergence term in code becomes:

```python
KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

**Explanation:**

- $ \log \sigma^2_{q,i} = \text{logvar} $
- $ \sigma^2_{q,i} = \exp(\text{logvar}) $
- The term inside the sum matches the simplified KL divergence formula.

---

### **4. Simplifying the Reconstruction Loss**

The reconstruction loss depends on the likelihood model $ p_\theta(\mathbf{x}|\mathbf{z}) $:

#### **a. Binary Cross-Entropy Loss**

For binary data and a Bernoulli likelihood:

- The reconstruction loss is the negative log-likelihood of the Bernoulli distribution.
- This corresponds to the binary cross-entropy (BCE) loss between the reconstructed data $ \hat{\mathbf{x}} $ and the original data $ \mathbf{x} $.

In code:

```python
BCE = nn.functional.binary_cross_entropy(
    x_reconstructed, x, reduction='sum')
```

#### **b. Mean Squared Error Loss**

For continuous data and a Gaussian likelihood:

- The reconstruction loss is the negative log-likelihood of the Gaussian distribution.
- This corresponds to the mean squared error (MSE) loss between $ \hat{\mathbf{x}} $ and $ \mathbf{x} $.

In code, you would use:

```python
MSE = nn.functional.mse_loss(
    x_reconstructed, x, reduction='sum')
```

---

### **5. Putting It All Together**

The total loss function becomes:

$$
L_\text{VAE} = \text{Reconstruction Loss} + \text{KL Divergence}
$$

In code:

```python
def loss_function(x_reconstructed, x, mu, logvar):
    # Reconstruction loss
    Reconstruction_Loss = ...

    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return Reconstruction_Loss + KLD
```

---

### **6. Why Use the Simplified Loss Function?**

#### **a. Computational Tractability**

- The theoretical loss function involves expectations and KL divergences that may be intractable or computationally expensive to compute.
- By making assumptions about the distributions (e.g., Gaussians with diagonal covariance), we can derive closed-form expressions for these terms.

#### **b. Analytical Solutions for KL Divergence**

- The KL divergence between two Gaussians can be computed analytically when the covariance matrices are diagonal.
- This avoids the need for numerical integration or sampling to estimate the KL divergence.

#### **c. Efficient Implementation**

- The simplified loss function can be efficiently computed using vectorized operations in libraries like PyTorch.
- This is crucial for training deep neural networks on large datasets.

#### **d. Practical Performance**

- Despite the simplifications, this loss function works well in practice and allows the VAE to learn meaningful latent representations.
- The balance between reconstruction accuracy and regularization is maintained.

---

### **7. Connection to the Theoretical Loss Function**

Let's connect the simplified loss function back to the theoretical one:

#### **a. Reconstruction Loss**

- **Theoretical Expression:**
  - $$\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \left[ \log p_\theta(\mathbf{x}|\mathbf{z}) \right]$$
- **Practical Implementation:**
  - For each data point, we sample $ \mathbf{z} $ from $ q_\phi(\mathbf{z}|\mathbf{x}) $ using the reparameterization trick.
  - Compute $ \hat{\mathbf{x}} = p_\theta(\mathbf{x}|\mathbf{z}) $.
  - Compute the reconstruction loss using BCE or MSE.

#### **b. KL Divergence**

- **Theoretical Expression:**
  $$
  D_\text{KL}( q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}) )
  $$
- **Practical Implementation:**
  - Compute the KL divergence analytically using the expressions derived for Gaussians.
  - Use the code provided to compute this term.

---

### **8. The Reparameterization Trick**

To compute the expected reconstruction loss, we need to sample $ \mathbf{z} $ from $ q_\phi(\mathbf{z}|\mathbf{x}) $. However, sampling operations are not differentiable, which poses a problem for backpropagation.

**Solution:** The reparameterization trick.

- **Idea:** Express $ \mathbf{z} $ as a deterministic function of $ \boldsymbol{\mu}_\phi(\mathbf{x}) $, $ \boldsymbol{\sigma}_\phi(\mathbf{x}) $, and a random noise $ \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) $.
- **Formulation:**
  $$
  \mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}
  $$
- **Benefit:** Allows gradients to flow through $ \boldsymbol{\mu}_\phi $ and $ \boldsymbol{\sigma}_\phi $ during backpropagation.

---

### **9. Summary**

- **Theoretical Loss Function:**
  - Derived from variational inference principles.
  - Involves expectations and KL divergences that may be intractable.

- **Practical Loss Function:**
  - Makes specific assumptions about the distributions involved (e.g., Gaussian distributions with diagonal covariance).
  - Uses analytical expressions for the KL divergence.
  - Implements the reconstruction loss using appropriate loss functions (BCE or MSE).
  - Employs the reparameterization trick for efficient and differentiable sampling.

- **Why Simplify:**
  - **Computational Efficiency:** Simplified expressions are computationally efficient and suitable for training neural networks.
  - **Analytical Tractability:** Allows us to compute the loss function exactly without resorting to approximations or numerical integration.
  - **Practical Effectiveness:** Despite the simplifications, the VAE trained with this loss function performs well in practice.

---

## **Conclusion**

In practical VAE implementations, we simplify the theoretical loss function to a form that is computationally tractable and efficient while retaining the essential components that allow the model to learn meaningful latent representations. The simplified loss function you provided:

- Captures the key aspects of the theoretical loss function.
- Is derived under reasonable assumptions about the distributions involved.
- Allows for efficient training using modern deep learning frameworks.

****





# A Beginner's Guide to Variational Methods: Mean-Field Approximation

Variational Bayeisan (VB) Methods are a family of techniques that are very popular in statistical Machine Learning. VB methods allow us to re-write *statistical inference* problems (i.e. infer the value of a random variable given the value of another random variable) as *optimization* problems (i.e. find the parameter values that minimize some objective function).

This inference-optimization duality is powerful because it allows us to use the latest-and-greatest optimization algorithms to solve statistical Machine Learning problems (and vice versa, minimize functions using statistical techniques).

This post is an introductory tutorial on Variational Methods. I will derive the optimization objective for the simplest of VB methods, known as the Mean-Field Approximation. This objective, also known as the *Variational Lower Bound*, is exactly the same one used in [Variational Autoencoders](https://arxiv.org/abs/1312.6114) (a neat paper which I will explain in a follow-up post).

## Table of Contents

1. Preliminaries and Notation
2. Problem formulation
3. Variational Lower Bound for Mean-field Approximation
4. Forward KL vs. Reverse KL
5. Connections to Deep Learning

## Preliminaries and Notation


This article assumes that the reader is familiar with concepts like random variables, probability distributions, and expectations. [Here's a refresher](https://www.khanacademy.org/math/probability/random-variables-topic) if you forgot some stuff. Machine Learning & Statistics notation isn't standardized very well, so it's helpful to be really precise with notation in this post:

- Uppercase $X$denotes a random variable
- Uppercase $P(X)$ denotes the probability distribution over that variable
- Lowercase $x \sim P(X)$ denotes a value $x$ sampled (∼) from the probability distribution $P(X)$ via some generative process.
- Lowercase $p(X)$ is the density function of the distribution of $X$. It is a scalar function over the measure space of $X$.
- $p(X=x)$ (shorthand $p(x)$) denotes the density function evaluated at a particular value $x$. 

Many academic papers use the terms "variables", "distributions", "densities", and even "models" interchangeably. This is not necessarily wrong per se, since $X$, $P(X)$, and $p(X)$ all imply each other via a one-to-one correspondence. However, it's confusing to mix these words together because their types are different (it doesn't make sense to *sample* a function, nor does it make sense to *integrate* a distribution). 

We model systems as a collection of random variables, where some variables ($X$) are "observable", while other variables ($Z$) are "hidden". We can draw this relationship via the following graph:

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-18 下午2.28.05.png" alt="截屏2024-09-18 下午2.28.05" style="zoom:50%;" />

The edge drawn from $Z$ to $X$ relates the two variables together via the conditional distribution $P(X|Z)$.

Here's a more concrete example: $X$ might represent the "raw pixel values of an image", while $Z$ is a binary variable such that $Z=1$ "if $X$ is an image of a cat".

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-18 下午2.30.15.png" alt="截屏2024-09-18 下午2.30.15" style="zoom:50%;" />

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-18 下午2.31.35.png" alt="截屏2024-09-18 下午2.31.35" style="zoom:33%;" />

[Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes'_theorem) gives us a general relationship between any pair of random variables:
$$
p(Z|X) = \frac{p(X|Z)p(Z)}{p(X)}
$$
The various pieces of this are associated with common names:

$p(Z|X)$ is the **posterior probability**: "given the image, what is the probability that this is of a cat?" If we can sample from $z\sim P(Z|X)$, we can use this to make a cat classifier that tells us whether a given image is a cat or not.

$p(X|Z)$ is the **likelihood**: "given a value of $Z$ this computes how "probable" this image $X$ is under that category ({"is-a-cat" / "is-not-a-cat"}). If we can sample from $x\sim P(X|Z)$, then we generate images of cats and images of non-cats just as easily as we can generate random numbers. If you'd like to learn more about this, see my other articles on generative models: [[1$](http://blog.evjang.com/2016/06/generative-adversarial-nets-in.html), [[2$](http://blog.evjang.com/2016/06/understanding-and-implementing.html).

$p(Z)$ is the **prior probability**. This captures any prior information we know about $Z$ - for example, if we think that $1/3$ of all images in existence are of cats, then $p(Z=1)=\frac{1}{3} ~\text{and} ~p(Z=0)=\frac{2}{3}$.

### Hidden Variables as Priors


*This is an aside for interested readers. Skip to the [next section](http:/#aproblem) to continue with the tutorial.*

The previous cat example presents a very conventional example of observed variables, hidden variables, and priors. However, it's important to realize that the distinction between hidden / observed variables is somewhat arbitrary, and you're free to factor the graphical model however you like.

We can re-write Bayes' Theorem by swapping the terms:
$$
\frac{p(Z|X)p(X)}{p(Z)} = p(X|Z)
$$
The "posterior" in question is now $P(X|Z)$

Hidden variables can be interpreted from a [Bayesian Statistics](https://en.wikipedia.org/wiki/Bayesian_statistics) framework as *prior beliefs* attached to the observed variables. For example, if we believe $X$ is a multivariate Gaussian, the hidden variable $Z$ might represent the mean and variance of the Gaussian distribution. The distribution over parameters $P(Z)$ is then a *prior* distribution to $P(X)$.

You are also free to choose which values $X$ and $Z$ represent. For example, $Z$ could instead be "mean, cube root of variance, and $X+Y$ where $Y \sim \mathcal{N}(0,1)$". This is somewhat unnatural and weird, but the structure is still valid, as long as $P(X|Z)$ is modified accordingly.

You can even "add" variables to your system. The prior itself might be dependent on other random variables via $P(Z| \theta)$, which have prior distributions of their own $P(\theta)$, and those have priors still, and so on. Any hyper-parameter can be thought of as a prior. In Bayesian statistics, [it's priors all the way down](https://en.wikipedia.org/wiki/Turtles_all_the_way_down).

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-18 下午2.49.23.png" alt="截屏2024-09-18 下午2.49.23" style="zoom:50%;" />

## Problem Formulation

The key problem we are interested in is *posterior inference*, or computing functions on the hidden variable $Z$. Some canonical examples of posterior inference:

- *Given this surveillance footage $X$, did the suspect show up in it?*
- *Given this twitter feed $X$, is the author depressed?*
- *Given historical stock prices $X_{1:t−1}$, what will $X_t$ be?*

We usually assume that we know how to compute functions on likelihood function $P(X|Z)$ and priors $P(Z)$.

The problem is, for complicated tasks like above, we often don't know how to sample from $P(Z|X)$ or compute $p(X|Z)$. Alternatively, we might know the form of $p(Z|X)$, but the corresponding computation is so complicated that we cannot evaluate it in a reasonable amount of time. We could try to use sampling-based approaches like [MCMC](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo), but these are slow to converge.

## Variational Lower Bound for Mean-field Approximation

The idea behind variational inference is this: let's just perform inference on an easy, parametric distribution $Q_{\phi}(Z|X)$ (like a Gaussian) for which we know how to do posterior inference, but adjust the parameters $\phi$ so that $Q_{\phi}$ is as close to $P$ as possible.

This is visually illustrated below: the blue curve is the true posterior distribution, and the green distribution is the variational approximation (Gaussian) that we fit to the blue density via optimization.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-18 下午3.00.21.png" alt="截屏2024-09-18 下午3.00.21" style="zoom:33%;" />

What does it mean for distributions to be "close"? Mean-field variational Bayes (the most common type) uses the Reverse KL Divergence to as the distance metric between two distributions.
$$
KL(Q_\phi(Z|X)||P(Z|X)) = \sum_{z \in Z}{q_\phi(z|x)\log\frac{q_\phi(z|x)}{p(z|x)}}
$$
Reverse KL divergence measures the amount of information (in nats, or units of $\frac{1}{log(2)}$ bits) required to "distort" $P(Z)$ into $Q_{\phi}(Z)$. We wish to minimize this quantity with respect to $\phi$.

By definition of a conditional distribution, $p(z|x)=\frac{p(x,z)}{p(x)}$. Let's substitute this expression into our original KL expression, and then distribute:
$$
\begin{align} 
KL(Q||P) & = \sum_{z \in Z}{q_\phi(z|x)\log\frac{q_\phi(z|x)p(x)}{p(z,x)}} && \text{(1)} \\ 
& = \sum_{z \in Z}{q_\phi(z|x)\big(\log{\frac{q_\phi(z|x)}{p(z,x)}} + \log{p(x)}\big)} \\ 
& = \Big(\sum_{z}{q_\phi(z|x)\log{\frac{q_\phi(z|x)}{p(z,x)}}}\Big) + \Big(\sum_{z}{\log{p(x)}q_\phi(z|x)}\Big) \\ 
& = \Big(\sum_{z}{q_\phi(z|x)\log{\frac{q_\phi(z|x)}{p(z,x)}}}\Big) + \Big(\log{p(x)}\sum_{z}{q_\phi(z|x)}\Big) && \text{note: $\sum_{z}{q(z)} = 1 $} \\ 
& = \log{p(x)} + \Big(\sum_{z}{q_\phi(z|x)\log{\frac{q_\phi(z|x)}{p(z,x)}}}\Big)  \\ 
\end{align}
$$
To minimize $KL(Q||P)$ with respect to variational parameters $\phi$, we just have to minimize $\displaystyle \sum_{z}{q_\phi(z|x)\log{\frac{q_\phi(z|x)}{p(z,x)}}}$, since $logp(x)$ is fixed with respect to $\phi$. Let's re-write this quantity as an expectation over the distribution $Q_{\phi}(Z|X)$.
$$
\begin{align} 
\sum_{z}{q_\phi(z|x)\log{\frac{q_\phi(z|x)}{p(z,x)}}} & = \mathbb{E}_{z \sim Q_\phi(Z|X)}\big[\log{\frac{q_\phi(z|x)}{p(z,x)}}\big]\\ 
& = \mathbb{E}_{z \sim Q_\phi(Z|X)}\big[ \log{q_\phi(z|x)} - \log{p(x,z)} \big] \\ 
& = \mathbb{E}_{z \sim Q_\phi(Z|X)}\big[ \log{q_\phi(z|x)} - (\log{p(x|z)} + \log(p(z))) \big] && \text{(via  $\log{p(x,z)=p(x|z)p(z)}$) }\\ 
& = \mathbb{E}_{z \sim Q_\phi(Z|X)}\big[ \log{q_\phi(z|x)} - \log{p(x|z)} - \log(p(z))) \big] \\ 
\end{align} \\
$$

> # **Why we could use Expectation to represent the summation?**
>
> ### **Background**
>
> In your derivation, you're working with the Kullback-Leibler (KL) divergence between two probability distributions $ q_\phi(z|x) $ and $ p(z, x) $. Specifically, you've reached the following expression:
>
> $
> \text{KL}(Q||P) = \sum_{z} q_\phi(z|x) \log \frac{q_\phi(z|x)}{p(z, x)}
> $
>
> Your question is:
>
> **Why can we write this summation as an expectation?**
>
> $
> \sum_{z} q_\phi(z|x) \log \frac{q_\phi(z|x)}{p(z, x)} = \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{q_\phi(z|x)}{p(z, x)} \right]
> $
>
> ### **Answer**
>
> The reason we can rewrite the summation as an expectation is due to the definition of the expectation of a function with respect to a probability distribution.
>
> #### **Definition of Expectation**
>
> For a discrete random variable $ z $ with probability mass function $ q(z) $, the expectation of a function $ f(z) $ is defined as:
>
> $
> \mathbb{E}_{z \sim q(z)} [f(z)] = \sum_{z} q(z) f(z)
> $
>
> Similarly, for a continuous random variable with probability density function $ q(z) $, the expectation is:
>
> $
> \mathbb{E}_{z \sim q(z)} [f(z)] = \int q(z) f(z) \, dz
> $
>
> #### **Applying to Your Expression**
>
> In your case, $ q(z) = q_\phi(z|x) $ and $ f(z) = \log \frac{q_\phi(z|x)}{p(z, x)} $.
>
> So, the summation:
>
> $
> \sum_{z} q_\phi(z|x) \log \frac{q_\phi(z|x)}{p(z, x)}
> $
>
> is **by definition** the expectation of $ f(z) $ under the distribution $ q_\phi(z|x) $:
>
> $
> \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{q_\phi(z|x)}{p(z, x)} \right]
> $
>
> #### **Why This Works**
>
> - **Summation as Expectation**: The expectation operator $ \mathbb{E}_{z \sim q(z)} $ is essentially a weighted sum (or integral) where each term $ f(z) $ is weighted by the probability $ q(z) $.
>
> - **Discrete Variables**: For discrete random variables, the expectation is a sum over all possible values of $ z $:
>
>   $
>   \mathbb{E}_{z \sim q(z)} [f(z)] = \sum_{z} q(z) f(z)
>   $
>
> - **Continuous Variables**: For continuous random variables, the expectation is an integral:
>
>   $
>   \mathbb{E}_{z \sim q(z)} [f(z)] = \int q(z) f(z) \, dz
>   $
>
> - **Your Expression**: Since you're summing over $ z $ and weighting $ \log \frac{q_\phi(z|x)}{p(z, x)} $ by $ q_\phi(z|x) $, this matches the definition of an expectation.
>
> #### **Conclusion**
>
> Therefore, you can rewrite the summation as an expectation because:
>
> - **Definition Alignment**: The form $ \sum_{z} q(z) f(z) $ is exactly how the expectation $ \mathbb{E}_{z \sim q(z)} [f(z)] $ is defined.
>
> - **Clarity and Notation**: Writing the expression as an expectation simplifies notation and makes it clearer that we're averaging $ \log \frac{q_\phi(z|x)}{p(z, x)} $ over the distribution $ q_\phi(z|x) $.
>
> ### **Further Explanation with an Example**
>
> Let's consider a simple example to illustrate this concept.
>
> #### **Example**
>
> Suppose $ z $ can take on values $ \{ z_1, z_2, z_3 \} $, and $ q(z) $ assigns probabilities:
>
> - $ q(z_1) = 0.2 $
> - $ q(z_2) = 0.5 $
> - $ q(z_3) = 0.3 $
>
> Let $ f(z) = \log \frac{q(z)}{p(z, x)} $.
>
> Then:
>
> $
> \sum_{z} q(z) f(z) = q(z_1) f(z_1) + q(z_2) f(z_2) + q(z_3) f(z_3)
> $
>
> This sum is precisely the expectation $ \mathbb{E}_{z \sim q(z)} [f(z)] $.
>
> ### **Application in Variational Inference**
>
> In the context of Variational Autoencoders (VAEs) and variational inference:
>
> - **Objective**: We aim to approximate the true posterior $ p(z|x) $ with a tractable distribution $ q_\phi(z|x) $.
>
> - **KL Divergence**: The KL divergence between $ q_\phi(z|x) $ and $ p(z|x) $ is:
>
>   $
>   \text{KL}(q_\phi(z|x) || p(z|x)) = \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{q_\phi(z|x)}{p(z|x)} \right]
>   $
>
> - **Variational Lower Bound**: We often manipulate expressions involving expectations to derive the Evidence Lower Bound (ELBO), which we optimize during training.
>
> ### **Summary**
>
> - **Key Point**: The summation $ \sum_{z} q(z) f(z) $ is equivalent to the expectation $ \mathbb{E}_{z \sim q(z)} [f(z)] $ by definition.
>
> - **Reason**: This equivalence holds because both represent the weighted average of the function $ f(z) $ with respect to the probability distribution $ q(z) $
>

****

Minimizing this is equivalent to *maximizing* the negation of this function:
$$
\begin{align} 
\text{maximize } \mathcal{L} & = -\sum_{z}{q_\phi(z|x)\log{\frac{q_\phi(z|x)}{p(z,x)}}} \\ 
& = \mathbb{E}_{z \sim Q_\phi(Z|X)}\big[ -\log{q_\phi(z|x)} + \log{p(x|z)} + \log(p(z))) \big] \\ 
& =  \mathbb{E}_{z \sim Q_\phi(Z|X)}\big[ \log{p(x|z)} + \log{\frac{p(z)}{ q_\phi(z|x)}} \big] && \text{(2)} \\ 
\end{align}
$$
In literature, $\mathcal{L}$ is known as the *variational lower bound*, and is computationally tractable if we can evaluate $p(x|z),~p(z),~q(z|x)$. We can further re-arrange terms in a way that yields an intuitive formula:
$$
\begin{align*} 
\mathcal{L} & =  \mathbb{E}_{z \sim Q_\phi(Z|X)}\left[ \log{p(x|z)} + \log{\frac{p(z)}{ q_\phi(z|x)}} \right] \\ 
& =   \mathbb{E}_{z \sim Q_\phi(Z|X)}\left[ \log{p(x|z)} \right] + \mathbb{E}_{q_\phi(z|x)}\left[ \log{\frac{p(z)}{ q_\phi(z|x)}} \right] && \text{Linearity of expectation} \\ 
& =   \mathbb{E}_{z \sim Q_\phi(Z|X)}\big[ \log{p(x|z)} \big] + \sum_{z}{q(z|x)\log{\frac{p(z)}{ q_\phi(z|x)}}} && \text{Definition of expectation} \\ 
& =  \mathbb{E}_{z \sim Q_\phi(Z|X)}\left[ \log{p(x|z)} \right] - \text{KL}(q_\phi(z|x)\,||\,p(z)) && \text{Definition of KL divergence} && \text{(3)}
\end{align*}
$$

If sampling $z∼Q(Z|X)$ is an "encoding" process that converts an observation $x$ to latent code $z$, then sampling $x∼Q(X|Z)$ is a "decoding" process that reconstructs the observation from $z$.

It follows that $\mathcal{L}$ is the sum of the expected "decoding" likelihood (how good our variational distribution can decode a sample of $Z$ back to a sample of $X$), plus the KL divergence between the variational approximation and the prior on $Z$. If we assume $Q(Z|X)$ is conditionally Gaussian, then prior $Z$ is often chosen to be a diagonal Gaussian distribution with mean 0 and standard deviation 1.

Why is $\mathcal{L}$ called the variational lower bound? Substituting $\mathcal{L}$ back into Eq. (1), we have:
$$
\begin{align*} 
KL(Q||P) & = \log p(x) - \mathcal{L} \\ 
\log p(x) & = \mathcal{L} + KL(Q||P) && \text{(4)} 
\end{align*}
$$
The meaning of Eq. (4), in plain language, is that $p(x)$, the log-likelihood of a data point $x$ under the true distribution, is $\mathcal{L}$, plus an error term $KL(Q||P)$ that captures the distance between $Q(Z|X=x)$ and $P(Z|X=x)$ at that particular value of $X$.

Since $KL(Q||P)≥0$, $\log p(x)$ must be greater than $\mathcal{L}$. Therefore $\mathcal{L}$ is a *lower bound* for $logp(x)$. L is also referred to as evidence lower bound (ELBO), via the alternate formulation:
$$
\mathcal{L} = \log p(x) - KL(Q(Z|X)||P(Z|X)) = \mathbb{E}_Q\big[ \log{p(x|z)} \big] - KL(Q(Z|X)||P(Z))
$$
Note that $\mathcal{L}$ itself contains a KL divergence term between the approximate posterior and the prior, so there are two KL terms in total in $\log p(x)$.

## Forward KL vs. Reverse KL

KL divergence is *not* a symmetric distance function, i.e. $KL(P||Q)≠KL(Q||P)$ (except when $Q≡P$) The first is known as the "forward KL", while the latter is "reverse KL". So why do we use Reverse KL? This is because the resulting derivation would require us to know how to compute $p(Z|X)$, which is what we'd like to do in the first place.

I really like Kevin Murphy's explanation in the [PML textbook](https://www.cs.ubc.ca/~murphyk/MLbook/), which I shall attempt to re-phrase here:

Let's consider the forward-KL first. As we saw from the above derivations, we can write KL as the expectation of a "penalty" function $\log \frac{p(z)}{q(z)}$ over a weighing function $p(z)$.
$$
\begin{align*} 
KL(P||Q) & = \sum_z p(z) \log \frac{p(z)}{q(z)} \\ 
& = \mathbb{E}_{p(z)}{\big[\log \frac{p(z)}{q(z)}\big]}\\ 
\end{align*}
$$
The penalty function contributes loss to the total KL wherever $p(Z)>0$. For $p(Z)>0$, $\lim_{q(Z) \to 0} \log \frac{p(z)}{q(z)} \to \infty$. This means that the forward-KL will be large wherever $Q(Z)$ fails to "cover up" $P(Z)$.

Therefore, the forward-KL is minimized when we ensure that $q(z)>0$ wherever $p(z)>0$. The optimized variational distribution $Q(Z)$ is known as "zero-avoiding" (density avoids zero when $p(Z)$ is zero).

> # Why forward-KL does not work
>
> 1. **Expectation Over $ P(z) $:**
>    - The forward KL divergence is an expectation over the true distribution $ P(z) $.
>    - This means the divergence focuses on where $ P(z) $ has significant probability mass.
>
> 2. **Penalty Function $ \log \frac{P(z)}{Q(z)} $:**
>
>    - The term $ \log \frac{P(z)}{Q(z)} $ acts as a penalty.
>    - If $ Q(z) $ is much less than $ P(z) $ (especially if $ Q(z) \to 0 $ while $ P(z) > 0 $), the penalty becomes large (approaches infinity).
>
> 3. **Zero-Avoiding Behavior:**
>
>    - The forward KL divergence heavily penalizes situations where $ Q(z) $ assigns low probability to regions where $ P(z) $ has high probability.
>    - Therefore, minimizing $ D_{\text{KL}}(P || Q) $ encourages $ Q(z) $ to be non-zero wherever $ P(z) $ is non-zero.
>    - This behavior is called **zero-avoiding** because $ Q(z) $ avoids being zero where $ P(z) $ is positive.
>
> 4. **Implications for Optimization:**
>
>    - When using forward KL divergence, the optimized $ Q(z) $ tends to "cover" all regions where $ P(z) $ has support.
>    - This can lead to $ Q(z) $ assigning probability mass to areas that $ P(z) $ considers unlikely, just to avoid the penalty.
>
> #### **Why This Matters:**
>
> - In variational inference, we are approximating the true distribution $ P(z) $ (often intractable) with a simpler distribution $ Q(z) $.
> - We need an objective function to measure how close $ Q(z) $ is to $ P(z) $.
> - The choice between forward and reverse KL divergence affects the behavior of the approximation.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-21 上午12.35.58.png" alt="截屏2024-09-21 上午12.35.58" style="zoom:50%;" />

Minimizing the Reverse-KL has exactly the opposite behavior:
$$
\begin{align*} 
KL(Q||P) & = \sum_z q(z) \log \frac{q(z)}{p(z)} \\ 
& = \mathbb{E}_{q(z)}{\big[\log \frac{q(z)}{p(z)}\big]} 
\end{align*}
$$

> # Why reverse KL works
>
> #### **Reverse KL Divergence $ D_{\text{KL}}(Q || P) $:**
>
> 1. **Expectation Over $ Q(z) $:**
>
>    - The reverse KL divergence is an expectation over $ Q(z) $.
>    - This means it focuses on where $ Q(z) $ assigns significant probability mass.
>
> 2. **Penalty Function $ \log \frac{Q(z)}{P(z)} $:**
>
>    - If $ P(z) $ is much less than $ Q(z) $ (especially if $ P(z) \to 0 $ while $ Q(z) > 0 $), the penalty becomes large.
>    - It penalizes $ Q(z) $ for assigning probability to regions where $ P(z) $ is low.
>
> 3. **Zero-Forcing Behavior:**
>
>    - Minimizing $ D_{\text{KL}}(Q || P) $ encourages $ Q(z) $ to assign zero probability to regions where $ P(z) $ is zero or very low.
>    - $ Q(z) $ focuses on high-probability regions of $ P(z) $.

<img src="../../Library/Application Support/typora-user-images/截屏2024-09-21 上午12.51.40.png" alt="截屏2024-09-21 上午12.51.40" style="zoom:50%;" />

So in summary, minimizing forward-KL "stretches" your variational distribution $Q(Z)$ to cover **over** the entire $P(Z)$ like a tarp, while minimizing reverse-KL "squeezes" the $Q(Z)$ **under** $P(Z)$.

It's important to keep in mind the implications of using reverse-KL when using the mean-field approximation in machine learning problems. If we are fitting a unimodal distribution to a multi-modal one, we'll end up with more false negatives (there is actually probability mass in $P(Z)$ where we think there is none in $Q(Z)$).

---

# The entire process of derive the Loss Function

## **1. Notation**

Let's establish the notation we'll use throughout the derivation:

- **Data Variables**:
  - $ \mathbf{x} \in \mathbb{R}^D $: Observed data vector (e.g., an image).
  - $ \mathbf{z} \in \mathbb{R}^L $: Latent (hidden) variable vector.

- **Probability Distributions**:
  - $ p_\theta(\mathbf{x}, \mathbf{z}) $: Joint distribution of data and latent variables, parameterized by $ \theta $.
  - $ p_\theta(\mathbf{x} | \mathbf{z}) $: Likelihood of data given latent variables (decoder).
  - $ p(\mathbf{z}) $: Prior distribution over latent variables (e.g., standard normal $ \mathcal{N}(\mathbf{0}, \mathbf{I}) $).
  - $ p_\theta(\mathbf{x}) $: Marginal likelihood of data under the model.
  - $ p_\theta(\mathbf{z} | \mathbf{x}) $: True posterior distribution of latent variables given data.
  - $ q_\phi(\mathbf{z} | \mathbf{x}) $: Approximate posterior distribution (encoder), parameterized by $ \phi $.

- **Objective**:
  - Our goal is to maximize the marginal likelihood $ p_\theta(\mathbf{x}) $ with respect to the parameters $ \theta $, but this is intractable due to the integral over $ \mathbf{z} $.

---

## **2. The Problem**

The marginal likelihood (evidence) of the data $ \mathbf{x} $ under the model is:

$$
p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x}, \mathbf{z}) \, d\mathbf{z} = \int p_\theta(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}
$$
Direct computation of $ p_\theta(\mathbf{x}) $ is often intractable due to the high-dimensional integral over $ \mathbf{z} $.

---

## **3. Introducing the Variational Lower Bound (ELBO)**

To make the problem tractable, we introduce an approximate posterior distribution $ q_\phi(\mathbf{z} | \mathbf{x}) $ and derive a lower bound on $ \log p_\theta(\mathbf{x}) $.

### **Step 1: Start with the Log Marginal Likelihood**

$$
\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}
$$



### **Step 2: Introduce the Approximate Posterior $ q_\phi(\mathbf{z} | \mathbf{x}) $**

We can rewrite the log marginal likelihood by multiplying and dividing by $ q_\phi(\mathbf{z} | \mathbf{x}) $:

$$
\log p_\theta(\mathbf{x}) = \log \int q_\phi(\mathbf{z} | \mathbf{x}) \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \, d\mathbf{z}
$$


### **Step 3: Apply Jensen's Inequality**

Since the logarithm is a concave function, we can apply Jensen's inequality to obtain a lower bound:

$$
\log p_\theta(\mathbf{x}) \geq \int q_\phi(\mathbf{z} | \mathbf{x}) \log \left( \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \right) \, d\mathbf{z}
$$
This integral is the expectation over $ q_\phi(\mathbf{z} | \mathbf{x}) $:

$$
\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} \left[ \log \left( \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \right) \right]
$$
The right-hand side is known as the Evidence Lower BOund (ELBO).

---

## **4. Expanding the ELBO**

Let's expand the ELBO to make it more interpretable.

### **Step 4: Expand the Joint Distribution**

Recall that the joint distribution $ p_\theta(\mathbf{x}, \mathbf{z}) $ can be expressed as:

$$
p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} | \mathbf{z}) p(\mathbf{z})
$$


### **Step 5: Substitute Back into the ELBO**

Substitute $ p_\theta(\mathbf{x}, \mathbf{z}) $ into the ELBO:

$$
\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} \left[ \log \left( \frac{p_\theta(\mathbf{x} | \mathbf{z}) p(\mathbf{z})}{q_\phi(\mathbf{z} | \mathbf{x})} \right) \right]
$$


### **Step 6: Separate the Logarithm**

Using properties of logarithms, we can split the log of a product into a sum of logs:

$$
\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} \left[ \log p_\theta(\mathbf{x} | \mathbf{z}) + \log p(\mathbf{z}) - \log q_\phi(\mathbf{z} | \mathbf{x}) \right]
$$


---

## **5. Rearranging the ELBO**

Now, we can rearrange terms to separate the reconstruction term and the regularization term.

### **Step 7: Split the Expectation**

The expectation of a sum is the sum of expectations:

$$
\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} [ \log p_\theta(\mathbf{x} | \mathbf{z}) ] + \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} [ \log p(\mathbf{z}) ] - \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} [ \log q_\phi(\mathbf{z} | \mathbf{x}) ]
$$


### **Step 8: Recognize the KL Divergence**

The last two terms can be combined into a negative Kullback-Leibler (KL) divergence:

$$
\mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} [ \log p(\mathbf{z}) - \log q_\phi(\mathbf{z} | \mathbf{x}) ] = - D_{\text{KL}} ( q_\phi(\mathbf{z} | \mathbf{x}) \, || \, p(\mathbf{z}) )
$$


#### **Explanation**:

- **KL Divergence Definition**:
  $$
  D_{\text{KL}} ( q(\mathbf{z}) \, || \, p(\mathbf{z}) ) = \int q(\mathbf{z}) \log \left( \frac{q(\mathbf{z})}{p(\mathbf{z})} \right) \, d\mathbf{z}
  $$
  
- **Negative KL Divergence**:
  $$
  D_{\text{KL}} ( q(\mathbf{z}) \, || \, p(\mathbf{z}) ) = \int q(\mathbf{z}) \left( \log p(\mathbf{z}) - \log q(\mathbf{z}) \right) \, d\mathbf{z}
  $$
  

### **Step 9: Write the ELBO in Terms of Reconstruction and Regularization**

Now, the ELBO becomes:

$$
\mathcal{L} = \underbrace{ \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} [ \log p_\theta(\mathbf{x} | \mathbf{z}) ] }_{\text{Reconstruction Term}} - \underbrace{ D_{\text{KL}} ( q_\phi(\mathbf{z} | \mathbf{x}) \, || \, p(\mathbf{z}) ) }_{\text{Regularization Term}}
$$


---

## **6. Interpretation of the ELBO Components**

### **Reconstruction Term**

- **Expression**:
$$
  \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} [ \log p_\theta(\mathbf{x} | \mathbf{z}) ]
$$


- **Meaning**:

  - Measures how well the model can reconstruct the data $ \mathbf{x} $ from the latent representation $ \mathbf{z} $.
  - Encourages the decoder $ p_\theta(\mathbf{x} | \mathbf{z}) $ to produce outputs close to the original data when given latent variables sampled from $ q_\phi(\mathbf{z} | \mathbf{x}) $.

### **Regularization Term (KL Divergence)**

- **Expression**:

  $$
  D_{\text{KL}} ( q_\phi(\mathbf{z} | \mathbf{x}) \, || \, p(\mathbf{z}) )
  $$
  
- **Meaning**:

  - Measures how close the approximate posterior $ q_\phi(\mathbf{z} | \mathbf{x}) $ is to the prior $ p(\mathbf{z}) $.
  - Acts as a regularizer to prevent the latent space from diverging too much from the prior distribution.
  - Encourages the encoder to produce latent representations that are consistent with the prior, facilitating sampling and generative capabilities.

---

## **7. Final ELBO Objective**

The ELBO provides a lower bound on the log marginal likelihood:

$$
\log p_\theta(\mathbf{x}) \geq \mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x})} [ \log p_\theta(\mathbf{x} | \mathbf{z}) ] - D_{\text{KL}} ( q_\phi(\mathbf{z} | \mathbf{x}) \, || \, p(\mathbf{z}) )
$$
Our objective during training is to maximize the ELBO with respect to both $ \theta $ and $ \phi $:

$$
\theta^*, \phi^* = \arg\max_{\theta, \phi} \mathcal{L}(\theta, \phi; \mathbf{x})
$$


---

## **8. Summary of Each Step**

Let's revisit each step with explanations:

1. **Notation Setup**: Defined all variables and distributions to ensure clarity in the derivation.

2. **Problem Identification**: Recognized that computing $ p_\theta(\mathbf{x}) $ directly is intractable due to the integral over $ \mathbf{z} $.

3. **Introducing $ q_\phi(\mathbf{z} | \mathbf{x}) $**: Multiplied and divided the integrand by $ q_\phi(\mathbf{z} | \mathbf{x}) $ to facilitate the application of Jensen's inequality.

4. **Applying Jensen's Inequality**: Used the concavity of the logarithm to derive a lower bound (ELBO) on $ \log p_\theta(\mathbf{x}) $.

5. **Expanding the Joint Distribution**: Substituted $ p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) $ into the ELBO.

6. **Separating the Logarithm**: Split the logarithm of the product into the sum of logarithms to isolate terms.

7. **Splitting the Expectation**: Used linearity of expectation to separate the terms.

8. **Recognizing the KL Divergence**: Combined the last two terms into the KL divergence between $ q_\phi(\mathbf{z} | \mathbf{x}) $ and $ p(\mathbf{z}) $.

9. **Writing the Final ELBO**: Presented the ELBO in terms of the reconstruction and regularization terms, which are interpretable and computable.

---

## **9. Additional Insights**

### **Why Use Variational Inference?**

- **Intractable Posteriors**: The true posterior $ p_\theta(\mathbf{z} | \mathbf{x}) $ is often complex and intractable.
- **Approximate Posterior $ q_\phi(\mathbf{z} | \mathbf{x}) $**: Introduces a tractable distribution (e.g., Gaussian) to approximate the true posterior.
- **Optimization Goal**: By maximizing the ELBO, we indirectly minimize the KL divergence between $ q_\phi(\mathbf{z} | \mathbf{x}) $ and $ p_\theta(\mathbf{z} | \mathbf{x}) $, bringing the approximate posterior closer to the true posterior.

### **Reparameterization Trick**

- **Challenge**: Backpropagation through stochastic nodes (sampling $ \mathbf{z} $) is problematic.
- **Solution**: Use the reparameterization trick to express $ \mathbf{z} $ as a deterministic function of $ \mathbf{x} $ and a noise variable $ \boldsymbol{\epsilon} $ (e.g., $ \mathbf{z} = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon} $, where $ \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) $).
- **Benefit**: Allows gradients to flow through $ \mu_\phi $ and $ \sigma_\phi $ during optimization.

---

## **10. Practical Implementation**

### **Encoder Network ($ q_\phi(\mathbf{z} | \mathbf{x}) $)**

- Outputs parameters of the approximate posterior (e.g., mean $ \mu_\phi(\mathbf{x}) $ and variance $ \sigma_\phi^2(\mathbf{x}) $).
- Typically modeled as a neural network.

### **Decoder Network ($ p_\theta(\mathbf{x} | \mathbf{z}) $)**

- Models the likelihood of the data given the latent variables.
- Generates reconstructions $ \hat{\mathbf{x}} $ from $ \mathbf{z} $.

### **Loss Function**

- **Total Loss**:

  $$
  \mathcal{L}_{\text{total}} = -\mathcal{L} = \text{Reconstruction Loss} + \text{KL Divergence}
  $$
  
- **Reconstruction Loss**:

  - For continuous data: Mean Squared Error (MSE) or negative log-likelihood assuming Gaussian likelihood.
  - For binary data: Binary Cross-Entropy (BCE) loss assuming Bernoulli likelihood.

- **KL Divergence Term**:

  - Computed analytically for certain distributions (e.g., between two Gaussians).



# Implementing VAE

- Defining the VAE architecture (encoder and decoder)
- Implementing the reparameterization trick
- Defining the loss function (combining reconstruction loss and KL divergence)
- Setting up the training loop
- Using the MNIST dataset for demonstration

Let's get started.

---

## **1. Import Necessary Libraries**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
```

- **`torch`**: Core PyTorch library.
- **`torch.nn`**: Contains neural network layers and loss functions.
- **`torch.optim`**: Optimization algorithms.
- **`torchvision.datasets`**: Common datasets for vision tasks.
- **`torchvision.transforms`**: Data augmentation and preprocessing utilities.

---

## **2. Define Hyperparameters**

```python
# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 20  # Dimensionality of the latent space
```

---

## **3. Prepare the Dataset**

We'll use the MNIST dataset for this example.

```python
# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download and load the training data
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)
```

---

## **4. Define the VAE Model**

### **Encoder**

```python
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(400, latent_dim)     # Mean of the latent distribution
        self.fc_logvar = nn.Linear(400, latent_dim)  # Log-variance of the latent distribution

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
```

### **Decoder**

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 28 * 28)
        self.sigmoid = nn.Sigmoid()  # To output values between 0 and 1

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_reconstructed = self.sigmoid(self.fc2(h))
        x_reconstructed = x_reconstructed.view(-1, 1, 28, 28)  # Reshape back to image size
        return x_reconstructed
```

### **VAE Model Combining Encoder and Decoder**

```python
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)     # Sample from standard normal
        z = mu + eps * std              # Sample from N(mu, var)
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar
```

---

## **5. Define the Loss Function**

The loss function combines the reconstruction loss and the KL divergence.

```python
def loss_function(x_reconstructed, x, mu, logvar):
    # Reconstruction loss (binary cross-entropy)
    BCE = nn.functional.binary_cross_entropy(
        x_reconstructed, x, reduction='sum')

    # KL divergence between the learned latent distribution and standard normal distribution
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    return BCE + KLD
```

---

## **6. Initialize the Model, Optimizer, and Device**

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

---

## **7. Training Loop**

```python
model.train()
for epoch in range(1, num_epochs + 1):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        x_reconstructed, mu, logvar = model(data)
        loss = loss_function(x_reconstructed, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item() / len(data):.4f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
```

- **Explanation:**
  - **`model.train()`**: Sets the model in training mode.
  - **Loop over epochs and batches**: Iterates through the dataset multiple times.
  - **Forward pass**: Compute the reconstructed output and latent variables.
  - **Compute loss**: Using the defined `loss_function`.
  - **Backward pass**: Computes gradients.
  - **Optimizer step**: Updates the model parameters.
  - **Logging**: Prints out the loss every 100 batches.

---

## **8. Save and Visualize Results**

### **Saving the Model**

```python
torch.save(model.state_dict(), 'vae_model.pth')
```

### **Visualizing Reconstructed Images**

```python
import matplotlib.pyplot as plt

# Function to display images
def imshow(img):
    img = img.squeeze().cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()

# Get a batch of test data
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=16, shuffle=True)

# Set model to evaluation mode
model.eval()
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        x_reconstructed, _, _ = model(data)
        break  # Only need one batch

# Display original images
print("Original Images")
grid_img = torchvision.utils.make_grid(data.cpu(), nrow=4)
imshow(grid_img)

# Display reconstructed images
print("Reconstructed Images")
grid_img = torchvision.utils.make_grid(x_reconstructed.cpu(), nrow=4)
imshow(grid_img)
```

- **Explanation:**
  - **`model.eval()`**: Sets the model to evaluation mode (e.g., disables dropout).
  - **Visualize original and reconstructed images**: Helps to assess the performance of the VAE.

---

## **9. Generating New Samples**

One of the exciting features of VAEs is the ability to generate new data by sampling from the latent space.

```python
# Sample from the latent space and generate new images
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)  # Sample from standard normal
    generated = model.decoder(z)

# Display generated images
print("Generated Images")
grid_img = torchvision.utils.make_grid(generated.cpu(), nrow=4)
imshow(grid_img)
```

- **Explanation:**
  - **Sampling from the latent space**: We sample random vectors from a standard normal distribution.
  - **Decoding**: Use the decoder to generate images from these latent vectors.
  - **Visualizing generated images**: See how well the VAE can generate new, plausible digits.

---

## **Complete Code**

For convenience, here's the complete code consolidated:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 10
latent_dim = 20  # Dimensionality of the latent space

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download and load the training data
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Define Encoder
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 28 * 28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.fc1(z))
        x_reconstructed = self.sigmoid(self.fc2(h))
        x_reconstructed = x_reconstructed.view(-1, 1, 28, 28)
        return x_reconstructed

# Define VAE
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

# Loss function
def loss_function(x_reconstructed, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(
        x_reconstructed, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model and optimizer
model = VAE(latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
model.train()
for epoch in range(1, num_epochs + 1):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        x_reconstructed, mu, logvar = model(data)
        loss = loss_function(x_reconstructed, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item() / len(data):.4f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

# Save the model checkpoint
torch.save(model.state_dict(), 'vae_model.pth')

# Function to display images
def imshow(img):
    img = img.squeeze().cpu().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()

# Load test data
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=16, shuffle=True)

# Visualize reconstructed images
model.eval()
with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        x_reconstructed, _, _ = model(data)
        break  # Only need one batch

print("Original Images")
grid_img = torchvision.utils.make_grid(data.cpu(), nrow=4)
imshow(grid_img)

print("Reconstructed Images")
grid_img = torchvision.utils.make_grid(x_reconstructed.cpu(), nrow=4)
imshow(grid_img)

# Generate new samples
with torch.no_grad():
    z = torch.randn(16, latent_dim).to(device)
    generated = model.decoder(z)

print("Generated Images")
grid_img = torchvision.utils.make_grid(generated.cpu(), nrow=4)
imshow(grid_img)
```

---

## **Explanation and Notes**

- **Encoder and Decoder Architecture:**
  - The encoder reduces the dimensionality of the input data and outputs parameters for the latent distribution (mean and log-variance).
  - The decoder reconstructs the data from the latent representation.

- **Reparameterization Trick:**
  - Essential for backpropagation through stochastic variables.
  - Allows gradients to flow through $ \mu $ and $ \log\sigma^2 $ by expressing $ z $ as a deterministic function of $ \epsilon $, $ \mu $, and $ \sigma $.

- **Loss Function Components:**
  - **Binary Cross-Entropy (BCE):** Measures the reconstruction error between the input and the output.
  - **KL Divergence (KLD):** Regularizes the latent space to be close to a standard normal distribution.

- **Training Process:**
  - The model learns to minimize the total loss (reconstruction + regularization).
  - The optimizer updates the parameters to improve both reconstruction quality and adherence to the prior distribution.

- **Visualization:**
  - Displaying original, reconstructed, and generated images helps assess the model's performance.

- **Device Configuration:**
  - The code automatically uses GPU acceleration if available, otherwise defaults to CPU.

---

## **Potential Extensions**

- **Hyperparameter Tuning:**
  - Experiment with different `latent_dim`, `learning_rate`, and `batch_size` values.
  - Adjust the architecture (e.g., adding more layers or neurons).

- **Different Datasets:**
  - Try using other datasets like CIFAR-10 or Fashion-MNIST.
  - Be mindful of adjusting the input/output dimensions accordingly.

- **Advanced Loss Functions:**
  - Incorporate more sophisticated loss functions or priors.
  - Implement beta-VAE by introducing a weight on the KL divergence term.

- **Conditional VAE (CVAE):**
  - Modify the model to include labels as additional inputs.
  - Allows the generation of data conditioned on specific classes.





## **Mathematical Explanation of the variance**

#### **Understanding Variance, Log-Variance, and Standard Deviation**

1. **Variance ($\sigma^2$) and Log-Variance ($\log \sigma^2$)**:

   - In the encoder of a VAE, we often predict the logarithm of the variance ($\log \sigma^2$) instead of the variance itself for numerical stability and to ensure positivity.
   - The variance is then recovered by exponentiating the log-variance:

     $
     \sigma^2 = e^{\log \sigma^2}
     $

2. **Standard Deviation ($\sigma$) and Variance ($\sigma^2$)**:

   - The standard deviation is the square root of the variance:

     $
     \sigma = \sqrt{\sigma^2}
     $

3. **Combining the Two Steps**:

   - To compute the standard deviation from the log-variance:

     $
     \sigma = \sqrt{\sigma^2} = \sqrt{e^{\log \sigma^2}} = e^{\frac{1}{2} \log \sigma^2}
     $

     This uses the logarithmic identity:

     $
     \sqrt{e^{a}} = e^{\frac{1}{2} a}
     $

---

### **Implementation in Code**

Given that `logvar` is $\log \sigma^2$, we can compute the standard deviation `std` as:

```python
std = torch.exp(0.5 * logvar)
```

This directly computes $ \sigma = e^{0.5 \log \sigma^2} $.

#### **Your Suggested Code**:

You proposed:

```python
std = torch.exp(logvar) ** 0.5
```

This can be rewritten as:

```python
variance = torch.exp(logvar)
std = torch.sqrt(variance)
```

Both methods aim to compute the standard deviation from the log-variance, but `torch.exp(0.5 * logvar)` is more efficient and numerically stable.

---

### **Detailed Derivation**

Let's derive why both methods are mathematically equivalent.

1. **Compute Variance from Log-Variance**:

   $
   \sigma^2 = e^{\log \sigma^2}
   $

2. **Compute Standard Deviation from Variance**:

   $
   \sigma = \sqrt{\sigma^2}
   $

3. **Combine Steps**:

   $
   \sigma = \sqrt{e^{\log \sigma^2}} = e^{\frac{1}{2} \log \sigma^2}
   $

   - This follows from the property of exponents:

     $
     \sqrt{e^{a}} = e^{a / 2}
     $

4. **Simplify**:

   $
   e^{\frac{1}{2} \log \sigma^2} = e^{\log \sigma^{1}} = \sigma
   $

   - Since:

     $
     \frac{1}{2} \log \sigma^2 = \log \sigma
     $

---

### **Why Use `torch.exp(0.5 * logvar)`?**

#### **Advantages**:

1. **Numerical Stability**:

   - Computing `torch.exp(logvar)` may result in large values if `logvar` is large, which can cause numerical overflow.
   - Multiplying `logvar` by 0.5 before exponentiating keeps the exponent smaller, reducing the risk of overflow.

2. **Computational Efficiency**:

   - Fewer operations are involved. Computing `torch.exp(logvar) ** 0.5` involves two exponentiations and a power operation, whereas `torch.exp(0.5 * logvar)` involves one exponentiation and a multiplication.

3. **Simpler Computational Graph**:

   - A simpler computational graph leads to more efficient backpropagation during training.

4. **Code Readability**:

   - The code directly reflects the mathematical expression for computing the standard deviation from the log-variance.

---

### **Example Calculation**

Let's verify with a numerical example.

#### **Assume**:

- `logvar = torch.tensor(2.0)`

#### **Using Your Method**:

```python
variance = torch.exp(logvar)        # variance = e^(2.0) = 7.3891
std = torch.sqrt(variance)          # std = sqrt(7.3891) = 2.7183
```

#### **Using `torch.exp(0.5 * logvar)`**:

```python
std = torch.exp(0.5 * logvar)       # std = e^(0.5 * 2.0) = e^(1.0) = 2.7183
```

#### **Result**:

- Both methods yield `std = 2.7183`.

---

### **Mathematical Equivalence**

To confirm the equivalence:

1. **Your Method**:

   $
   \sigma = \sqrt{e^{\log \sigma^2}} = e^{\frac{1}{2} \log \sigma^2}
   $

2. **Implemented Method**:

   $
   \sigma = e^{0.5 \times \log \sigma^2}
   $

- Both expressions are the same.

---

### **Summary**

- **Formula**:

  $
  \sigma = e^{0.5 \times \log \sigma^2}
  $

- **Implementation**:

  ```python
  std = torch.exp(0.5 * logvar)
  ```

- **Benefits**:

  - **Efficiency**: Fewer computations.
  - **Stability**: Reduced risk of numerical overflow.
  - **Simplicity**: Clear reflection of mathematical relationships.

---

### **Conclusion**

Using `std = torch.exp(0.5 * logvar)` is a mathematically sound and computationally efficient way to compute the standard deviation from the log-variance in the context of VAEs. It ensures numerical stability and simplifies the computational graph, which is advantageous during training.

---

### **Additional Note on the Reparameterization Trick**

In VAEs, the reparameterization trick is used to allow gradients to flow through stochastic variables. The standard deviation computed as `std = torch.exp(0.5 * logvar)` is essential for this.

The sampling of $ z $ is done as:

$
z = \mu + \sigma \cdot \epsilon
$

Where:

- $ \mu $ is the mean predicted by the encoder.
- $ \sigma $ is the standard deviation computed from `logvar`.
- $ \epsilon $ is sampled from a standard normal distribution $ \mathcal{N}(0, 1) $.

This allows the randomness to be isolated in $ \epsilon $, and since $ \sigma $ depends on `logvar`, gradients can flow through $ \mu $ and `logvar` during backpropagation.

---

I hope this explanation clarifies why `std = torch.exp(0.5 * logvar)` is used and how it corresponds to the mathematical relationships between variance, log-variance, and standard deviation. Let me know if you have any further questions!
