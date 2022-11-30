######################################################
# Yasmin Heimann, hyasmin, 311546915
# @output Image classifier model
# @description A module that trains and evaluates a convolutional neural network,
#              using various optimized hyper-parameters.
#              The module is trained and tested on an imbalanced CIFAR-10 images data sets.
#              The model accuracy reached 88% when use with the default parameters defined here.
######################################################

## IMPORT of packages ##
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import Counter
# ---- #
from dataset import get_dataset_as_torch_dataset, get_dataset_as_array
from models import SimpleModel
from Model_Evaluation import test_accuracy, test_per_class, test_existing_model

## PROGRAM GENERAL PARAMETERS ##
# Hyper-parameters #
BATCH_SIZE = 32
EPOCS = 100
LR_LIST = [0.001]
# Chooe the transformation to apply. for all put: [0,1,2,3,4], for only the first one: [0]
TRANSFORMATIONS_INDEX = []
MOMENTUM = 0.9
WEIGHT_DECAY = 0.001

# Program state indicators #
# Choose weather to use a weighted training loss or a weighted sampler to fix Imbalance in the Data Set
# Only one should be True each time
WEIGHTED_LOSS = True
SAMPLER = False

# Magic numbers #
LABEL_INDEX = 1
TRAIN_DATA_PATH = "./train.pickle"
TEST_DATA_PATH = "./dev.pickle"
SAVE_MODEL_PATH = "./model_311546915.ckpt"  # model added so it won't override the existing final model
LR_INDEX = 0
LR_LOSS = 1
EVAL_INDEX = 2


def plot_lr(data):
    """
    Plots a graph that shows the learning process given the loss in each epoc and a learning rate
    :param data: list contains LR, list of loss per epoc and the overall model evaluation
    """
    print(data)
    x_axis = list(range(1, EPOCS + 1))
    # apply multiple graphs if multiple LR were trained
    for d in data:
        label = (str(d[LR_INDEX]) + ", %.2f %%") % (d[EVAL_INDEX])
        plt.plot(x_axis, d[LR_LOSS], label=label)
    # set x axis label
    plt.xlabel('epocs')
    # Set the y axis label of the current axis.
    plt.ylabel('training loss')
    # Set a title of the current axes.
    title = 'Training Loss Over ' + str(EPOCS) + ' Epocs'
    plt.title(title)
    # show a legend on the plot
    plt.legend()
    # Display and save a figure.
    fig_name = "lr_convergence_" + str(EPOCS) + ".png"
    plt.savefig(fig_name, dpi=600)
    plt.show()


def check_epoc_acc(net, best_acc, test_loader):
    """
    Checks if the accuracy of the current epoc is better then the last maximum.
    If so, it replaces it and saves the current model
    :param net: the trained model
    :param best_acc: the highest accuracy achieved by now
    :param test_loader: the test set to evaluate the accuracy on
    :return: the best accuracy
    """
    cur_acc = test_per_class(net, test_loader, to_print=False)
    cur_best = best_acc
    if cur_acc > best_acc:
        cur_best = cur_acc
        net.save(SAVE_MODEL_PATH)
    return cur_best


def train_by_lr(net, trainloader, criterion, cur_lr, test_loader):
    """
    Train a neural network using PyTorch.
    :param net: Loaded Model
    :param trainloader: the train set to learn on
    :param criterion: the decision criterion - loss object
    :param cur_lr: the LR
    :param test_loader: the test set to evaluate each epoc
    :return: a list with the losses in each epod
    """
    optimizer = optim.Adam(net.parameters(), lr=cur_lr, weight_decay=WEIGHT_DECAY)
    epocs_loss = []
    best_acc = 0
    for epoc in range(EPOCS):  # loop over the dataset multiple times
        net.train()
        # initialize basic 0 loss
        running_loss = 0.0
        # run over the tensors batches in the data
        for i, data in enumerate(trainloader, 0):
            # get the inputs and labels from the data - a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # add current loss
            running_loss += loss.item()
        # print the average loss at the end of the epoc
        epoc_cur_loss = running_loss / len(trainloader)
        print("epoc ", (epoc + 1) , " loss ", epoc_cur_loss)
        epocs_loss.append(epoc_cur_loss)
        # save the weights if they are better than before
        best_acc = check_epoc_acc(net, best_acc, test_loader)
    print('Finished Training Successfully and maybe Happily')
    return epocs_loss


def create_transforms():
    """
    Creates transformations of the data
    :return: the transformations object
    """
    transformations = [transforms.ToPILImage(),
        transforms.RandomAffine(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    final_transforms = [transformations[i] for i in TRANSFORMATIONS_INDEX]
    return transforms.Compose(final_transforms)


def get_class_distribution():
    """
    Calculate the distribution of the classes in the train data set for fixing imbalances
    :return: a list with the number of samples per each class and a list of the true labels
    """
    data = get_dataset_as_array(TRAIN_DATA_PATH)
    labels = [item[LABEL_INDEX] for item in data]
    labels_count = Counter(labels)
    return [i for i in labels_count.values()], labels


def preprocess_data(trainset, class_count, labels):
    """
    Create the train set object
    :param trainset: the data set
    :param class_count: the districution of classes in the train data set
    :param labels: the true labels of the data set
    :return: the data object
    """
    # create randomly weighted batch sampler
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[labels]
    weighted_sampler = WeightedRandomSampler(weights=class_weights_all,
                                             num_samples=len(class_weights_all),
                                             replacement=True)
    # add transformations
    trainset.transform = create_transforms()
    # choose weather to use the sample or not
    if SAMPLER:
        return DataLoader(trainset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, sampler=weighted_sampler)
    return DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,)


def train_network():
    """
    Trains a neural network using the model given in models python file.
    """
    # prepare the train and test data
    class_distribution, labels = get_class_distribution()
    trainset = get_dataset_as_torch_dataset(TRAIN_DATA_PATH)
    trainloader = preprocess_data(trainset, class_distribution, labels)
    testset = get_dataset_as_torch_dataset(TEST_DATA_PATH)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,  # todo batch size?
                                              shuffle=False, num_workers=0)

    # create loss object
    criterion = nn.CrossEntropyLoss()
    if WEIGHTED_LOSS:
        normed_weights = [1 - (d / sum(class_distribution)) for d in class_distribution]
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(normed_weights))

    # learn
    lr_loss = []
    for lr in LR_LIST:
        # create the network object
        model = SimpleModel()
        # train the model
        train_loss = train_by_lr(model, trainloader, criterion, lr, test_loader)
        # evaluate the model
        eval = test_existing_model(SAVE_MODEL_PATH, to_print=True)
        lr_loss.append([lr, train_loss, eval])
    # plots the convergence rate per each LR examined
    plot_lr(lr_loss)


def test_network(trained_model):
    """
    Evaluates a trained model using general accuracy test and weighted class accuracy method
    :param trained_model: the model to evaluate
    :return: the weighted accuracy (take class distribution into account)
    """
    testset = get_dataset_as_torch_dataset(TEST_DATA_PATH)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=0)
    test_accuracy(trained_model, test_loader, to_print=False)
    return test_per_class(trained_model, test_loader, to_print=False)


train_network()


