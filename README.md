# LogSinkhornGPU

This package provides a GPU implementation of the Sinkhorn algorithm for L2 optimal transport with data living on rectangular 2D grids. 

The algorithm is grealy inspired by [geomloss](https://www.kernel-operations.io/geomloss/). As theirs, our implementation is _online_, this is, it does not store the cost matrix explicitly; and it uses the _separability trick_ for the squared cost, which reduces the complexity of the Sinkhorn iteration. Unlike theirs, our implementation is also fast on small problems (this is, input smaller than 64x64).The necessity and interest for a fast algorithm for small problems arises from the [domain decomposition algorithm](https://arxiv.org/abs/2001.10986), where the solution to a big problem is found by solving many small problems in parallel. For big problems we refer to geomloss, which has a greater and deeper scope than this package. 

## Installation

The installation follows the template in https://github.com/chrischoy/MakePytorchPlusPlus.

You must have `torch` installed.

```
git clone https://github.com/ismedina/LogSinkhornGPU
cd LogSinkhornGPU
python setup.py install
```

## Implementation

We describe briefly the separability trick that lowers the complexity of the sinkhorn iteration when $c$ is the squared cost on a regular grid. For input measures $\mu$ and $\nu$ and dual potentials $\alpha$ and $\beta$, the log-sinkhorn half-iteration reads: 

$$ \alpha_i =  - \varepsilon \log \sum_j \exp\left( h_j - \frac{|X_i - Y_j|^2_2}{\varepsilon}\right)$$

where $h_j := \nu_j + \beta_j / \varepsilon$ (obtaining $\beta$ from $\alpha$ is analogous). Since the data lives on respective rectangular grids of size $(M_1, M_2)$ and $(N_1, N_2)$, we can decompose the index $i$ into a pair of indices $i_1, i_2$, and analogously for $j$. Then, $X_{i_1, i_2} = (x_{i_1}, x_{i_2})$, and analogously for $Y$. With this indexing the iteration reads: 


$$ \alpha_{i_1, i_2} =  - \varepsilon \log \sum_{j_1, j_2} \exp\left( h_{j_1, j_2} - \frac{(x_{i_1} - y_{j_1})^2 + (x_{i_2} - y_{j_2})^2}{\varepsilon}\right)$$

or, more conveniently

$$ \alpha_{i_1, i_2} =  - \varepsilon \log 
\sum_{j_1} \exp\left(  - \frac{(x_{i_1} - y_{j_1})^2}{\varepsilon}\right)
\left[
\sum_{j_2}\exp\left( h_{j_1, j_2} - \frac{(x_{i_2} - y_{j_2})^2}{\varepsilon}
\right)
\right]
$$

Thus, the original logsumexp (with complexity in $O(M_1M_2N_1N_2)$) can be replaced by a first logsumexp (with complexity in $O(M_2N_1N_2)$)

$$ \tilde{\alpha}_{j_1, i_2} := \log \sum_{j_2}\exp\left( h_{j_1, j_2} - \frac{(x_{i_2} - y_{j_2})^2}{\varepsilon}\right) $$

followed by a second logsumexp (with complexity in $O(M_1M_2N_1)$):

$$ \alpha_{i_1, i_2} =  - \varepsilon \log 
\sum_{j_1} \exp\left(\tilde{\alpha}_{j_1, i_2}  - \frac{(x_{i_1} - y_{j_1})^2}{\varepsilon}\right)
$$

Finally, if the two grids start in the origin (this is, $x_{0,0} = y_{0,0} = (0,0)$ and have the same spacing between gridpoints $\Delta x$, then we can replace the cost term $(x_{i_1} - y_{j_1}) = (i_1 - j_1)\Delta x$. Thus, we don't need to pass the vectors $x$ and $y$ to the GPU kernel ---just the size of their grids and $\Delta x$---, minimizing the memory overhead. If the two grids do not start in the origin (as is usually the case in the domain decomposition algorithm), one has first to move the measures to this position, which introduces an offset to the duals that is easily computable. 
