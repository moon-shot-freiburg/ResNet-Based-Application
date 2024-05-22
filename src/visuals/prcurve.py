import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sys
import itertools

sys.path.append("../")

from models.resnet import ResNet18

def main():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_loader = data.DataLoader(
        datasets.CIFAR10(root='../models/data', train=False, transform=transforms.Compose([transforms.ToTensor(), normalize])),
        batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    resnet = ResNet18().cuda()
    resnet.load_state_dict(torch.load('../models/model_best.pth.tar')['state_dict'])
    resnet.eval()

    all_preds = []
    all_labels = []
    print('Validation started')

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            print('Batch [%4d / %4d]' % (i + 1, len(val_loader)))
            inputs, labels = input.cuda(), target.cuda()
            outputs = resnet(inputs)
            _, preds = torch.max(outputs.data, 1)
            all_preds.extend(preds.cpu().numpy())  # Use extend for list concatenation
            all_labels.extend(labels.cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    plot_confusion_matrix(cm, classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], normalize=True)
    plot_confusion_matrix(cm, classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], normalize=False)

    # PR curves
    plot_precision_recall_curves(all_preds, all_labels)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(8,8))
    plt.show()

def plot_precision_recall_curves(all_labels, all_preds):
    # Create a list to store precision-recall curves for all classes
    all_precisions = []
    all_recalls = []
    all_average_precisions = []

    # Loop through all classes
    for i in range(10):
        
        y_test_binary = np.copy(all_labels)  # Make a copy to avoid modifying original data
        #y_test_binary[y_test_binary != i] = 0  # Set non-matching labels to 0
        #y_test_binary[y_test_binary == i] = 1  # Set matching labels to 1
        # y_test_binary = (all_labels == i).astype(np.float64)  # Convert to binary for each class
        # all_preds_float = all_preds.astype(np.float64)
        # precision, recall, thresholds = precision_recall_curve(y_test_binary, all_preds == i)
        precision, recall, thresholds = precision_recall_curve(y_test_binary, all_preds)


        # Calculate average precision for this class
        average_precision = average_precision_score(y_test_binary, all_preds)
        all_average_precisions.append(average_precision)

        # Store precision and recall for later plotting
        all_precisions.append(precision)
        all_recalls.append(recall)

    # Plot PR curves for all classes
    plt.figure(figsize=(10, 8))
    for i, (precision, recall, average_precision) in enumerate(zip(all_precisions, all_recalls, all_average_precisions)):
        plt.plot(recall, precision, label=f'Class {i} (AUPRC={average_precision:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curves for ResNet18 on CIFAR-10 (Multi-class)')
    plt.legend(loc='lower left')
    plt.show()


if __name__ == "__main__":
    main()
