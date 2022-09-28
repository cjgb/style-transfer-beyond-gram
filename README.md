# Style transfer without/beyond Gram matrices

This repo provides code to discuss the relevance of
[Gram matrices](https://en.wikipedia.org/wiki/Gram_matrix) in
_style transfer_ (see
[L. Gatys et al](https://doi.org/10.48550/arXiv.1508.06576)
).

The paper itself does not provide much information about the actual role of Gram matrices in style transfer. They simply state that:

> To generate a texture that matches the style of a given image, we use gradient descent from a white noise image to find another image that matches the style representation of the original image. This is done by minimising the mean-squared distance between the entries of the Gram matrix from the original image and the Gram matrix of the image to be generated.

Likewise, most information you can get on the issue fail to provide an explanation of the role of Gram matrices in style transfer. In this repo, however, I am providing insights and code in order to cast a light on this matter.

## First things first

1. All code in this repo is based on ---and partially copy-pasted from--- the excellent book on Keras by François Chollet. In fact, it was while reading this book that I came with the idea of looking beyond the use of Gram matrices for style transfer.
2. In order to run the code, you'll need a Python installation with `pipenv`. You can install the required dependencies via `pipenv install` and then run the examples with `pipenv run python python/script_name.py`.

## What is style transfer? What is style?

[Style transfer](https://en.wikipedia.org/wiki/Neural_style_transfer) is a technology to _transfer_ the style of one image into another one. _Style_ is a term that we often find in discussions about art, mostly qualitative in nature, that would seem difficult to capture, measure, and, on top of that, operated upon. Therefore, the first hurdle in the creation of a technological tool for style transfer consists in defining _style_ in mathematical terms.

Style happens to be related to the statistical distribution of a collection (infinite in theory, of about a few hundreds in practice) of small features on an image. Some notes:

* Style is embodied in the full (joint) probabilistic distribution of these features, not in their marginal distributions.
* By small features one should understand something little patches. Eg, [pointillism](https://en.wikipedia.org/wiki/Pointillism) is a drawing technique that uses little points of colour; therefore, pointillist pictures will have patterns resembling a little circle in abundance. Other images will have large presence of little white patches, etc.
* It does not really matter where these patterns happen in the picture; what is relevant is their abundance and their _correlation_ ---here I understand correlation in a wide sense, not just in terms of, eg, Pearson's--- with other features.

Note also that what is stated above is a totally empirical claim based on the success of such operationalization of style in actually transferring style.

Then, after having selected a wide enough set of, say, N features, styles such as Picasso's, old Disney movies, Roman mosaics, etc. are characterized by a particular N-dimensional probability distribution.

Transfer style therefore consists in modifying one image so that its corresponding N-dimensional distribution matches a given target distribution (another style) without altering its high level features (shapes, etc.).


## Style and (convolutional) neural networks

Modern, complex, (convolutional) neural networks (nn in what follows), are well suited for the task of style transfer:

* The first layers in the nn capture a sufficiently large number of local patterns.
* The upper layers in the nn capture the general structure and shape of the represented image.

Say that your choice of nn has a set of properly selected _early layers_ `s_1`, `s_2`,... `s_n` capturing a sufficiently large number of local features and another properly selected number of _late layers_ `t_1`, `t_2`,... `t_m` which are sensitive to the overall shapes in an image.

Then, style transfer using such nn consists in creating a new image from a given one and a target style so that:

1. The m `t_i` layers from the original and the output image are similar (so that they represent the same _shapes_ globally, ie, that both represent, say, the same portrait of the same person).
2. The distribution of the values ---but not necessarily the values themselves--- of the `s_i` in the output and the target style images are similar.

Therefore, the loss function for style transfer would resemble something like:

`l(o, i, st) = sum_k(d1(t_k(o), t_k(i))) + sum_k(d2(s_k(o), s_k(st)))`

where:

* `o` is the output image
* `i` is the input image
* `st` is the image we want to borrow the style from
* `d1` is a _regular distance_, such as the square loss
* `d2` is a distance between multivariate probability distributions

## Distances between distributions

It is time to dissect the contents of the `s_i` layers. For a given image, the `s_i` layer is a 3 dimensional tensor whose dimensions represent:

1. A given local pattern. It is a relatively small patch, say, of size 5x5 pixels.
2. An X coordinate in the image.
3. A Y coordinate.

Each value in `s_i` therefore represents the abundance of a given pattern in a given position in the picture. If the layer captures, say, 100 different patterns and they have been measured in 1000 (X,Y) locations, the layer can be interpreted as a collection of 1000 samples of a 100-dimensional random variable.

In order to transfer style, we are required to find a distance between distributions represented by such samples. Statistics provides us with a number of alternatives:

* Moments. It is known that two distributions with equal moments (of all orders) have to be the same one. We could calculate all empirical moments (the vector of means, the covariance matrix, etc.) and get a metric out of it.
* [Kullback–Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
* [Maximum mean discrepancy](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions), usually known as MMD.
* Other?

The original paper used, as indicated above, the distance between the so called Gram matrices which is, in statistical terms, measuring the discrepancy in the second empirical moments (related to the covariance matrix). So it could be considered an approximation to the method of moments and also an application of a particular form of MMD.

It should be noted that measuring the discrepancy between the Gram matrices does not guarantee similarity between distributions. Two people with the same height and weight need not be the same person; likewise, two samples having similar Gram matrices may not come from similar distributions.

However, it can be shown in practice that measuring the discrepancy between distributions using the second moments (via Gram matrices) is _enough_ (and, also, less demanding computationally) than using a proper distance function.

Both KL and MMD (with certain kernels, such as the Gaussian) provide better guarantees of equality between distributions.

The purpose of this repo is to show it via example.

## Results
### KL distance

### MMD distance

## Final remarks

I consider a matter of bad taste using an approximation without clearly indicating it. Gatys el al should have said something along the following lines:

> We want to do A, but because of reasons X, Y, Z, we are going to apply an approximation via Gram matrices as others have done in I, J, and K.

They didn't and now many people are just doing _homeopatic_ machine learning: blindly reproducing a recipe without really knowing why. Fortunately, others have rushed to settle the score: see the aptly titled _Demystifying Neural Style Transfer_ reference below.

## References

* [Deep Learning with Keras](https://www.goodreads.com/book/show/42900232-deep-learning-with-keras), by F. Chollet.
* [A Neural Algorithm of Artistic Style](https://doi.org/10.48550/arXiv.1508.06576), by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
* [Demystifying Neural Style Transfer](https://doi.org/10.48550/arXiv.1701.01036), by Yanghao Li, Naiyan Wang, Jiaying Liu, and Xiaodi Hou.
* [Understanding style transfer](https://gcamp6f.com/2017/12/05/understanding-style-transfer/), by P.T.R. Rupprecht
