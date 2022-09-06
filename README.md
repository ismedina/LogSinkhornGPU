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

Test formula: 

$$ \log \sum_i \exp $$
