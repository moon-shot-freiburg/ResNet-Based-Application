from sklearn.metrics import confusion_matrix, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import itertools
sys.path.append("../")

from models.resnet import ResNet18

def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='../models/data', train=False, transform=transforms.Compose([transforms.ToTensor(), normalize,])), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    resnet = ResNet18().cuda()
    resnet.load_state_dict(torch.load('../models/model_best.pth.tar')['state_dict'])
    resnet.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []  # Store predicted probabilities for each class

    print('Validation started')
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            print('Batch [%4d / %4d]' % (i+1, len(val_loader)))
            inputs, labels = input.cuda(), target.cuda()
            outputs = resnet(inputs)
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.cpu().numpy())
            all_labels += list(labels.cpu().numpy())
            # Obtain predicted probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

        print("Shape of y_true:", len(all_labels))
        print("Example of y_true:", all_labels[:10])  

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    #plot_confusion_matrix(cm, classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], normalize=True)
    #plot_confusion_matrix(cm, classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], normalize=False)

    # Convert list of predicted probabilities into a numpy array
    all_probs = np.concatenate(all_probs)
    
    # Plot Precision-Recall Curve
    precision_recall_curve_plot(all_labels, all_probs)

def precision_recall_curve_plot(y_true, y_scores, n_classes=10):
    precision = dict()
    recall = dict()
    average_precision = dict()

    # Binarize the true labels
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))
    print("Shape of y_true_binarized:", y_true_binarized.shape)
    print("value of y_true_binarized:", y_true_binarized)

    print("Shape of y_scores:", y_scores.shape)
    print("value of y_scores:", y_scores[:5])
    print("value of y_scores:", np.sum(y_scores[5000]))

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], y_scores[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    # Compute micro-average precision-recall curve and its area
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_binarized.ravel(), y_scores.ravel())
    average_precision["micro"] = auc(recall["micro"], precision["micro"])

    # Plot Precision-Recall curve for each class
    plt.figure()
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'Class {i} (AP = {average_precision[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    main()
