
import torch
from collections import OrderedDict


def l2_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) -
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).pow(2).mean()


def dot_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return (
        torch.cat(tuple([t.view(-1) for t in dict_1_values])) *
        torch.cat(tuple([t.view(-1) for t in dict_2_values]))
    ).mean()


def sign_alignment_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    a = torch.sign(torch.cat(tuple([t.view(-1) for t in dict_1_values])))
    b = torch.sign(torch.cat(tuple([t.view(-1) for t in dict_2_values])))
    # count how many times they are the same
    return (a == b).sum().item() / len(a)


def rms_of_the_dict(dict_1):
    #  root mean square (RMS)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    # if there are nan values, replace them with 0
    for i, val in enumerate(dict_1_values):
        if torch.isnan(val).any():
            dict_1_values[i] = torch.zeros_like(val)
    return torch.cat(tuple([t.view(-1) for t in dict_1_values])).pow(2).mean().sqrt()


def norm_of_the_dict(dict_1):
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    # if there are nan values, replace them with 0
    for i, val in enumerate(dict_1_values):
        if torch.isnan(val).any():
            dict_1_values[i][torch.isnan(dict_1_values[i])] = 0
            # dict_1_values[i] = torch.zeros_like(val)
    return torch.norm(torch.cat(tuple([t.view(-1) for t in dict_1_values])))


def cos_between_dicts(dict_1, dict_2):
    assert len(dict_1) == len(dict_2)
    dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
    dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
    return torch.nn.functional.cosine_similarity(
        torch.cat(tuple([t.view(-1) for t in dict_1_values])),
        torch.cat(tuple([t.view(-1) for t in dict_2_values])),
        dim=0
    ).mean()


def average_between_dict_list(dict_list):
    assert len(dict_list) > 0
    dict_1 = dict_list[0]
    for dict_2 in dict_list[1:]:
        assert len(dict_1) == len(dict_2)
        dict_1_values = [dict_1[key] for key in sorted(dict_1.keys())]
        dict_2_values = [dict_2[key] for key in sorted(dict_1.keys())]
        dict_1 = OrderedDict([(key, val_1 + val_2) for key, val_1, val_2 in zip(
            sorted(dict_1.keys()), dict_1_values, dict_2_values)])
    dict_1 = OrderedDict([(key, val / len(dict_list)) for key, val in dict_1.items()])
    return dict_1


def get_grad_cos_sim(grad1, grad2):
    assert len(grad1) == len(grad2)
    cos_sim = torch.nn.functional.cosine_similarity(grad1.view(-1), grad2.view(-1), dim=0)
    return cos_sim


def get_grad_cos_batch(grad_batch):
    # compute the average of parwise cosine distance
    # grad_batch: (batch_size, num_params)
    cos = torch.nn.functional.cosine_similarity(grad_batch.unsqueeze(1),
                                                grad_batch.unsqueeze(0), dim=2)
    # remove the diagonal elements
    cos = cos[~torch.eye(cos.shape[0], dtype=torch.bool)].mean()
    return cos

