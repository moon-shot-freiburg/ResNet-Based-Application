from sklearn.metrics import confusion_matrix
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
    print('Validation started')
    
    with torch.no_grad():
        # for i, data in enumerate(val_loader):
        for i, (input, target) in enumerate(val_loader):
            #print('I am here')
            print('Batch [%4d / %4d]' % (i+1, len(val_loader)))
            inputs, labels = input.cuda(), target.cuda()
            outputs = resnet(inputs)
            _, preds = torch.max(outputs.data, 1)
            all_preds += list(preds.cpu().numpy())
            all_labels += list(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    plot_confusion_matrix(cm, classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], normalize=True)
    plot_confusion_matrix(cm, classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], normalize=False)



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

if __name__ == "__main__":
    main()