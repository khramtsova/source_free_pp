import copy
import numpy as np

import torch
from torch.nn import Identity
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from backpack import extend

from model import LeNet
from gmm_calculations import get_per_class_mean_and_cov, get_gmm_probab
from utils import load_features, GreyToColor, get_grad_norm_pytorch


class Model:
    def __init__(self, model_name, model_path,
                 device="cuda:0", num_classes=10):
        self.model_name = model_name
        self.device = device
        self.num_classes = num_classes
        self.feature_extractor, self.classifier = self.load_model(model_name, model_path)

        # We extend the classifier layer of the model to use the backpack library for the gradient computation
        self.classifier = extend(self.classifier)

    def load_model(self, model_name, model_path):
        """
        Load a model and initialise it with the weights from the model_path;
        Remove the last layer and return separately the feature extractor and the classifier

        Args:
            model_name:
            model_path:

        Returns:

        """
        if model_name == "lenet":
            net = LeNet(num_classes=self.num_classes)
            model_weights = torch.load(model_path)
            net.load_state_dict(model_weights, strict=False)
            net.n_outputs = 1024
            classifier = copy.deepcopy(net.last_layer)
            del net.last_layer
            net.last_layer = Identity()
        else:
            raise NotImplementedError

        return net, classifier


def predict_accuracy(classifier, feature_loader, num_classes=10, device="cuda:0"):
    """
    Predict the accuracy of the classifier using the method described in the paper
    Consists of 3 steps:
    1. Calculate the GMM statistics by calculating the mean and covariance of the logits for each class
    2. Calculate the probability of the features belonging to each class using the GMM statistics
    3. Make a prediction for each sample based on the gradient norm to the target class and the random class.
        If the gradient norm to the predicted class is higher than the gradient norm to the random class, the sample is predicted incorrectly
        If the gradient norm to the random class is higher than the gradient norm to the target class, the sample is predicted correctly
    Args:
        classifier: The last layer (Fully connected layer) of the model
        feature_loader: DataLoader iterating through the features
        num_classes: Number of classes in the dataset
        device: Device to run the computations on

    Returns: Ground truth accuracy, Predicted Accuracy, Mean Absolute Error between the two

    """

    classifier.eval()
    classifier.to(device)

    # Step 1: Calculate the GMM statistics
    gmm_stats = get_per_class_mean_and_cov(feature_loader, classifier, num_classes=num_classes)

    res = {"labels": [], "pred": [],  "weight2rand": [], "weight2target": []}

    for data in feature_loader:
        features, lbl = data[0].to(device), data[1].to(device)

        # Step 2: Adapt the probabilities based on the GMM statistics
        prob, logits = get_gmm_probab(classifier, features, num_classes, gmm_stats)
        pred_class = torch.argmax(logits, dim=1).detach().cpu()

        # Store the gradients
        uniform_dist = torch.ones_like(prob) / num_classes
        grad_weight_to_rand = get_grad_norm_pytorch(classifier, prob, uniform_dist)

        prob, logits = get_gmm_probab(classifier, features, num_classes, gmm_stats)
        pseudo_labels = torch.argmax(logits, dim=1)
        one_hot = torch.nn.functional.one_hot(pseudo_labels, num_classes=num_classes)
        # Log one_hot
        grad_weight_to_target = get_grad_norm_pytorch(classifier, prob, one_hot)

        # ========   Logs   =========
        # indx - index of a sample in a batch. Log each sample separately
        for indx, label in enumerate(lbl):
            res["labels"].append(label.item())
            res["pred"].append(pred_class[indx].item())
            res["weight2rand"].append(grad_weight_to_rand[indx].item())
            res["weight2target"].append(grad_weight_to_target[indx].item())

    # Ground Truth Accuracy
    gt_acc = sum(np.array(res["labels"]) == np.array(res["pred"])) / len(res["labels"])

    # Step 3: Predict the performance:
    #   The sample is predicted to be correctly classified
    #   if the gradient norm to the random class is higher than the gradient norm to the predicted class
    predicted_accuracy = sum(np.array(res["weight2rand"]) >= np.array(res["weight2target"])) / len(res["labels"])

    # MAE: Mean Absolute Error - the difference between the ground truth accuracy and the predicted accuracy
    mae = abs(gt_acc - predicted_accuracy)

    return gt_acc, predicted_accuracy, mae


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Define the model and split it into the feature extractor and the classifier
    model = Model(model_name="lenet", model_path="./logs/lenet_mnist.pt")
    # Now model has model.feature_extractor and model.classifier

    # Load the USPS dataset
    num_classes = 10
    batch_size = 16
    dsets = {}
    test_transform = transforms.Compose([transforms.Resize((32, 32)),
                                         transforms.ToTensor(),
                                         GreyToColor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ])
    dsets["usps"] = datasets.USPS(root="./logs/", train=False, download=True, transform=test_transform)
    dsets['svhn'] = datasets.SVHN(root="./logs/", split='test', download=True, transform=test_transform)

    for dset_name in dsets.keys():
        dataloader = DataLoader(dsets[dset_name], batch_size=32, shuffle=False, num_workers=0)


        # Pass the dataset through the feature extractor and store the features
        # If the features are already stored in a file, load them from the file
        feature_set = load_features(dataloader, model.feature_extractor, fname=f"./logs/{dset_name}_features.csv",
                                    device=device)

        # Create a DataLoader to iterate through the features
        feature_loader = DataLoader(feature_set, batch_size=batch_size, shuffle=True, num_workers=0)

        # Predict the accuracy using our method
        gt_acc, predicted_acc, mae = predict_accuracy(model.classifier, feature_loader,
                                                      num_classes=num_classes, device=device)

        print(f"Dataset: {dset_name} \n"
              f"GT Accuracy: {round(gt_acc* 100, 2)} "
              f"Predicted Accuracy: {round(predicted_acc*100, 2)} "
              f"MAE: {round(mae*100, 2)} \n")
    return

if __name__ == "__main__":
    main()




