import numpy as np
import torch
from torch.nn import functional as F


def poisson_kl(lambda1, lambda2):
    """
    Compute the KL divergence between two Poisson distributions.

    KL(λ1 || λ2) = λ2 - λ1 + λ1 * log(λ1 / λ2)

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (lambda1, lambda2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    lambda1 = torch.clamp(lambda1, min=1e-7)
    lambda2 = torch.clamp(lambda2, min=1e-7)

    return lambda2 - lambda1 + lambda1 * torch.log(lambda1 / lambda2)


def poisson_log_likelihood(x, *, lambdas):
    """
    Compute the log-likelihood of a Poisson distribution.

    log P(x|λ) = x * log(λ) - λ - log(x!)

    We omit the log(x!) term as it's constant w.r.t. λ and doesn't affect optimization.

    :param x: the observed counts (can be non-integer due to sampling).
    :param lambdas: the Poisson rate parameter Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    lambdas = torch.clamp(lambdas, min=1e-7)
    assert x.shape == lambdas.shape

    # Simplified Poisson log-likelihood (omitting constant log(x!) term)
    log_probs = x * torch.log(lambdas) - lambdas
    assert log_probs.shape == x.shape
    return log_probs


def binomial_kl(mean1, mean2):
    """
    Compute the KL divergence between two Bernoulli.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, mean2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"
    mean1mean2 = torch.clamp(mean1/(mean2 + 1e-7), min=1e-7)
    mean1mean2_r = torch.clamp((1 - mean1) / (1 - mean2 + 1e-7), min=1e-7)
    return mean1 * torch.log(mean1mean2) + (1 - mean1) * torch.log(mean1mean2_r)


def binomial_log_likelihood(x, *, means):
    """
    Compute the log-likelihood of a Binomial distribution.

    :param x: the binary mask.
    :param means: the Binomial mean Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    means = torch.clamp(means, min=1e-7, max=1-1e-7)
    assert x.shape == means.shape
    log_probs = x * torch.log(means) + (1 - x) * (torch.log(1 - means))
    assert log_probs.shape == x.shape
    return log_probs


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def focal_loss(inputs, targets, gamma=2):
    BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = (1-pt)**gamma * BCE_loss
    return F_loss.mean(dim=[1, 2, 3])
