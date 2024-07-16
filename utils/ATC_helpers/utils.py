import numpy as np


def softmax_numpy(preact):
    exponents = np.exp(preact)
    sum_exponents = np.sum(exponents, axis=1, keepdims=True)
    return exponents/sum_exponents


def inverse_softmax_numpy(preds):
    # preds[preds==0.0] = 1e-40
    preds = preds/np.sum(preds, axis=1, keepdims=True)
    return np.log(preds) - np.mean(np.log(preds),axis=1, keepdims=True)


def get_entropy(probs):
    return np.sum(np.multiply(probs, np.log(probs + 1e-20)), axis=1)


def find_threshold(scores, labels):
    sorted_idx = np.argsort(scores)

    sorted_scores = scores[sorted_idx]
    sorted_labels = labels[sorted_idx]

    fp = np.sum(labels == 0)
    fn = 0.0

    min_fp_fn = np.abs(fp - fn)
    thres = 0.0
    for i in range(len(labels)):
        if sorted_labels[i] == 0:
            fp -= 1
        else:
            fn += 1

        if np.abs(fp - fn) < min_fp_fn:
            min_fp_fn = np.abs(fp - fn)
            thres = sorted_scores[i]

    return min_fp_fn, thres

