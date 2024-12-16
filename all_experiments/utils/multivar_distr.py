import copy
import math
import torch
import numpy as np


def batch_mahalanobis(bL, bx):
    r"""
    Computes the squared Mahalanobis distance :math:`\mathbf{x}^\top\mathbf{M}^{-1}\mathbf{x}`
    for a factored :math:`\mathbf{M} = \mathbf{L}\mathbf{L}^\top`.

    Accepts batches for both bL and bx. They are not necessarily assumed to have the same batch
    shape, but `bL` one should be able to broadcasted to `bx` one.

    Taken from the official torch implementation:
    https://github.com/pytorch/pytorch/blob/main/torch/distributions/multivariate_normal.py
    """
    n = bx.size(-1)
    bx_batch_shape = bx.shape[:-1]

    # Assume that bL.shape = (i, 1, n, n), bx.shape = (..., i, j, n),
    # we are going to make bx have shape (..., 1, j,  i, 1, n) to apply batched tri.solve
    bx_batch_dims = len(bx_batch_shape)
    bL_batch_dims = bL.dim() - 2
    outer_batch_dims = bx_batch_dims - bL_batch_dims
    old_batch_dims = outer_batch_dims + bL_batch_dims
    new_batch_dims = outer_batch_dims + 2 * bL_batch_dims
    # Reshape bx with the shape (..., 1, i, j, 1, n)
    bx_new_shape = bx.shape[:outer_batch_dims]
    for sL, sx in zip(bL.shape[:-2], bx.shape[outer_batch_dims:-1]):
        bx_new_shape += (sx // sL, sL)
    bx_new_shape += (n,)
    bx = bx.reshape(bx_new_shape)
    # Permute bx to make it have shape (..., 1, j, i, 1, n)
    permute_dims = (
        list(range(outer_batch_dims))
        + list(range(outer_batch_dims, new_batch_dims, 2))
        + list(range(outer_batch_dims + 1, new_batch_dims, 2))
        + [new_batch_dims]
    )
    bx = bx.permute(permute_dims)

    flat_L = bL.reshape(-1, n, n)  # shape = b x n x n
    flat_x = bx.reshape(-1, flat_L.size(0), n)  # shape = c x b x n
    flat_x_swap = flat_x.permute(1, 2, 0)  # shape = b x n x c
    M_swap = (
        torch.linalg.solve_triangular(flat_L, flat_x_swap, upper=False).pow(2).sum(-2)
    )  # shape = b x c
    M = M_swap.t()  # shape = c x b

    # Now we revert the above reshape and permute operators.
    permuted_M = M.reshape(bx.shape[:-1])  # shape = (..., 1, j, i, 1)
    permute_inv_dims = list(range(outer_batch_dims))
    for i in range(bL_batch_dims):
        permute_inv_dims += [outer_batch_dims + i, old_batch_dims + i]
    reshaped_M = permuted_M.permute(permute_inv_dims)  # shape = (..., 1, i, j, 1)
    return reshaped_M.reshape(bx_batch_shape)


def log_likelihood_multivar(x, mean, cov, num_classes, L2_regularizer):
    # mu = µ.expand(x.shape + (-1,))

    mahal = batch_mahalanobis(cov, x - mean)
    if L2_regularizer:
         mahal = mahal / (torch.norm(cov, 2))
    # Variance: my_var = cov.pow(2).sum(-1)
    # This value is a constant
    k_ln_2pi = num_classes * math.log(2 * math.pi)


    # ln|Sigma|
    log_det = 2 * cov.diagonal(dim1=-2, dim2=-1).log().sum()
    # final_log_prob = - 0.5 * mahal
    final_log_prob = log_det - 0.5 * mahal + k_ln_2pi  # )

    # print(torch.norm(cov).shape, final_log_prob.shape)
    # print("Norm value",  torch.norm(cov))
    final_log_prob = final_log_prob # / torch.norm(cov)
    if L2_regularizer:
        final_log_prob = final_log_prob / torch.norm(cov, 2)
    return final_log_prob


def get_per_class_log_probs(mean_per_class, cov, num_classes, L2_regularizer=True):
    cov = torch.linalg.cholesky(cov)
    log_probs_means = {}
    # print(mean_per_class[0].shape, logits.shape)
    for cls in range(num_classes):
        if mean_per_class[cls] is not None:
            log_probs_means[cls] = []
            for cls1 in range(num_classes):
                if cls1 in mean_per_class.keys():
                    if cls != cls1:
                        if mean_per_class[cls1] is not None:
                            log_probs_means[cls].append(log_likelihood_multivar(mean_per_class[cls1],
                                                                    mean_per_class[cls],
                                                                    cov, num_classes,
                                                                    L2_regularizer=L2_regularizer))
                    #else:
                    #    # assign negative infinity to the log probability
                    #    log_probs_means[cls].append(torch.tensor(float("-100"), device=cov.device))
                #log_probs_means[cls].append(log_likelihood_multivar(mean_per_class[cls1], mean_per_class[cls],
                #                                                    cov, num_classes))
            log_probs_means[cls] = torch.stack(log_probs_means[cls], dim=0)
    log_probs_means = torch.stack(list(log_probs_means.values()), dim=0)
    return log_probs_means


def log_prob_logsumexp_trick(gmm_stats, log_probs_means, logits, num_classes, L2_regularizer=True,
                             count_per_class=None):
    log_probs, mahals = [], []
    cov, mean_per_class = gmm_stats["cov"], gmm_stats["mean_per_class"]
    # new_cov_shape = torch.broadcast_shapes(cov.shape[:-2], logits.shape[:-1])
    # cov_extended = cov.expand(new_cov_shape + (-1, -1))

    cov = torch.linalg.cholesky(cov)
    # cov_extended = torch.linalg.cholesky(cov_extended)
    for cls in range(num_classes):
        if cls in mean_per_class.keys():
            log_probs.append(log_likelihood_multivar(logits, mean_per_class[cls], cov, num_classes,
                                                     L2_regularizer=L2_regularizer))
    # mahals.append(batch_mahalanobis(cov, logits - mean_per_class[cls]))
        # else:
        #     log_probs.append(log_likelihood_multivar(logits, 0, cov, num_classes))
        #     mahals.append(batch_mahalanobis(cov_extended, logits - 0))

    log_probs = torch.stack(log_probs, dim=0)
    # mahal_total = torch.stack(mahals, dim=0)
    # print("log probs before", log_probs)

    log_probs = log_probs - torch.logsumexp(log_probs_means, dim=1).unsqueeze(1)
    # print("log probs after", log_probs.shape)
    # for indx, log_p in enumerate(log_probs):
    #     # cls 64 samples of class 1
    #     print(log_p.shape)
    #     logp_cls_excluded = torch.cat((log_p[:indx], log_p[indx+1:]), dim=0)
    #     print( torch.logsumexp(logp_cls_excluded, dim=0))
    #     log_probs[indx] = log_p - torch.logsumexp(logp_cls_excluded, dim=0)
    # print( torch.logsumexp(log_probs, dim=0).shape)
    # print(log_probs.shape)
    # raise
    log_probs = log_probs - torch.logsumexp(log_probs, dim=0).unsqueeze(0)

    # print("log probs after logsoftmax within", log_probs)
    # print("Sum of log probs", print(torch.logsumexp(log_probs, dim=1).unsqueeze(1)))
    # print(torch.logsumexp(log_probs, dim=1).unsqueeze(1))
    #
    # print("Log probs shape", log_probs.shape)
    # print("Mahals shape", mahal_total.shape)
    # raise
    #
    #
    log_exp = torch.exp(log_probs)
    # print("Log exp", log_exp)
    total_probs = torch.sum(log_exp, dim=0)
    prob = final_probabilities = (log_exp / total_probs).T
    # print("Final prob", prob)
    # print("PROB", prob, sum(prob[0]))
    #
    # # There is an error somewhere here
    #
    #
    # pi_pow_k = pow((2 * math.pi), num_classes)
    # det = torch.linalg.det(cov)
    # max_constant = max(mahal_total.flatten() * (-0.5)) - 70
    #
    # # value_1 = torch.log(1 / torch.sqrt(pi_pow_k * det))  #
    # value_1 = torch.log(1/torch.sqrt(det * pi_pow_k))  #
    # value_2 = max_constant + torch.log(torch.sum(torch.exp(-0.5 * mahal_total - max_constant), dim=0))
    # print(log_probs.shape, value_2.shape)
    # log_sum = value_2 + value_1
    # print(log_probs.shape, log_sum.shape)
    # final_log_probs = (log_probs - log_sum).T
    # print("Final log probs shape", final_log_probs.shape)
    try:
        assert torch.allclose(torch.sum(final_probabilities, dim=1),
                              torch.ones(final_probabilities.shape[0], device=final_probabilities.device),
                          rtol=1e-4)
    except AssertionError:
        print("PROBABILITIES DO NOT SUM TO 1")
        print("Logits", logits)
        print("Final probabilities", final_probabilities)
        raise AssertionError
    return final_probabilities



def logits_to_prob_multivar(multivar_normal, logits, num_classes):
    # New logits
    # Subtract the maximum value for numerical stability along each row
    # logits = logits - torch.max(logits, dim=1, keepdim=True)[0]

    #logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    #print("Logits after", logits)
    #print("Max logit", torch.max(logits, dim=1, keepdim=True)[0])

    normal_dist = []
    for cls in range(num_classes):
        if cls not in multivar_normal.keys():
            # For classes where the covariance matrix is not positive definite
            #   assign zero probability
            # for each class the shape is [1, batch_size]
            # normal_dist.append(torch.zeros([1, logits.shape[0]], device=device))
            raise ValueError("Covariance matrix is not positive definite for class {}".format(cls))
        else:
            # multivar_norm = torch.exp(multivar_normal[cls].log_prob(logits + 1e-9)).reshape(1, -1)
            log_prob = multivar_normal[cls].log_prob(logits)

            # Shift the log probabilities to avoid overflow using logsumexp
            normal_dist.append(log_prob)
            # normal_dist.append(multivariate_normal.pdf(logits_without_temp.cpu().detach().numpy(),
            #                                        mean=mean_per_class[cl],
            #                                        cov=cov_per_class[cl]).reshape(1, -1))
    # normal_dist = np.concatenate(normal_dist, axis=0)
    # total_probs = np.sum(normal_dist, axis=0)
    # assert np.allclose(np.sum(final_probabilities, axis=1), 1.0, rtol=1e-7)
    normal_dist = torch.stack(normal_dist)
    # print(normal_dist)
    # print(torch.sum(normal_dist, dim=1))
    # print(torch.logsumexp(normal_dist, dim=1).unsqueeze(1))
    #
    # print(torch.log(torch.sum(normal_dist, dim=1)).unsqueeze(1))

    # Perform logsumexp trick to avoid overflow
    normal_dist = normal_dist - torch.log(torch.sum(normal_dist, dim=1)).unsqueeze(1)

    # Perform logsumexp trick to avoid overflow
    # scaler = torch.logsumexp(normal_dist, dim=1)
    # normal_dist = normal_dist - scaler.unsqueeze(1)
    # scaler = torch.logsumexp(normal_dist, dim=0)
    # normal_dist = normal_dist - scaler.unsqueeze(0)

    normal_dist = torch.exp(normal_dist)

    total_probs = torch.sum(normal_dist, dim=0)
    prob = final_probabilities = (normal_dist / total_probs).T
    # try catch assert
    try:
        assert torch.allclose(torch.sum(final_probabilities, dim=1),
                              torch.ones(final_probabilities.shape[0], device=final_probabilities.device),
                          rtol=1e-4)
    except AssertionError:
        print("PROBABILITIES DO NOT SUM TO 1")
        print("Logits", logits)
        print("Final probabilities", final_probabilities)
        print("Normal Distr", normal_dist)
        raise AssertionError
    # make the probabilities back to torch tensor
    # prob = torch.tensor(prob).cuda()
    return prob

