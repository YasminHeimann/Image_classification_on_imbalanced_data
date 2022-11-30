######################################################
# Yasmin Heimann, hyasmin, 311546915
# @output Accuracy decrease rate for different epsilons and 5 perturbated images,
#         with the original image and the moise added.
# @description A module that generates adverarial examples,
#              from a given trained neural network in PyTorch.
######################################################

# IMPORTS #
from models import SimpleModel
import torch
from dataset import get_dataset_as_torch_dataset
from sklearn.metrics import confusion_matrix

# CHANGE THE PATH TO ADJUST YOUR MODEL AND TEST SET #
BASELINE_MODEL_PATH = "./pre_trained.ckpt"
TRAIN_DATA_PATH = "./train.pickle"
TEST_DATA_PATH = "./dev.pickle"
SAVED_MODEL_PATH = "./311546915.ckpt"

# SPECIFIC DATA PROPERTIES #
CLASS_NUM = 3
CLASSES = ('car', 'truck', 'cat')


def test_per_class(net, testloader, to_print=False):
    """
    Test the total accuracy of the model, taking into account data set imbalance.
    The function prints the accuracy per class, and the average accuracy overall.
    :param net: the trained model
    :param testloader: the test set to predict on
    :param to_print: indicates if to print the results or not
    :return: the final accuracy
    """
    class_correct = list(0. for i in range(CLASS_NUM))
    class_total = list(0. for i in range(CLASS_NUM))
    net.eval()  # evaluation mode turns off dropout layers
    with torch.no_grad():  # block gradient calculations as we are on evaluation mode
        for data in testloader:
            image, label = data
            outputs = net(image)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == label).squeeze()
            class_correct[label] += c.item()
            class_total[label] += 1
    # get the average loss on all classes
    overall_loss = 0
    for i in range(CLASS_NUM):
        class_loss = 100 * class_correct[i] / class_total[i]
        if to_print:
            print('Accuracy of %5s : %2d %%' % (CLASSES[i], class_loss))
        overall_loss += class_loss
    overall_loss /= CLASS_NUM
    # print results
    if to_print:
        print('Weighted Accuracy is %2d %%' % (overall_loss))
    return overall_loss


def test_accuracy(net, testloader, to_print=False):
    """
    Test the total accuracy of the model without taking into account data set imbalance.
    :param net: the trained model
    :param testloader: the test set to predict on
    :param to_print: indicates if to print the results or not
    :return: the final accuracy
    """
    correct, total, c = 0, 0, 0
    net.eval()  # evaluation mode turns off dropout layers
    with torch.no_grad():  # block gradient calculations as we are on evaluation mode
        for data in testloader:
            c+=1
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # calculate the accuracy percentage
    acc = 100 * correct / total
    if to_print:
        # prints results
        print("Number of Tested images: ", c)
        print('Accuracy of the network on the test images: %d %%' % (acc))
    return acc


def model_confusion_matrix(model, testloader):
    """
    Calculates a confusion matrix for a given model and test set
    """
    correct, total = 0, 0
    predictions, true_labels = [], []
    model.eval()  # evaluation mode turns off dropout layers
    with torch.no_grad():  # block gradient calculations as we are on evaluation mode
        # go over the data and count the predictions and true labels
        for i, data in enumerate(testloader):
            images, labels = data
            outputs = model(images)
            _ , predicted = torch.max(outputs.data, 1)
            predictions.extend(list(predicted.numpy()))
            true_labels.extend(list(labels.numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # use sklearn confusion matrix function
    c_matrix = confusion_matrix(true_labels, predictions)
    c_matrix = c_matrix / c_matrix.sum(axis=1).reshape(1, 3)
    # calculate the accuracy percentage
    acc = 100 * ((c_matrix[0, 0] + c_matrix[1, 1] + c_matrix[2, 2]) / CLASS_NUM)
    # prints results
    print("Accuracy: " + str(acc) + "%")
    print(confusion_matrix(true_labels, predictions))


def test_confusion_matrix(path=SAVED_MODEL_PATH):
    """
    Prints the confusion matrix of a given model
    :param path: a path to the trained model to calculate its matrix
    """
    # load the model and the test data set
    net = SimpleModel()
    net.load(path)
    testset = get_dataset_as_torch_dataset(TEST_DATA_PATH)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
                                              shuffle=False, num_workers=0)
    # calculate the model confusion matrix and prints it
    model_confusion_matrix(net, test_loader)


def test_existing_model(model_path=SAVED_MODEL_PATH, data_set_path=TEST_DATA_PATH, to_print=True):
    """
    Test a saved model. Prints the its accuracy on each class, and total accuracy
    """
    # load the model and the test data set
    net = SimpleModel()
    net.load(model_path)
    testset = get_dataset_as_torch_dataset(data_set_path)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,  # todo batch size?
                                              shuffle=False, num_workers=0)
    # calculate accuracy of the model, with and without respect to class imbalance
    test_accuracy(net, test_loader, to_print)
    return test_per_class(net, test_loader, to_print)


def pre_trained():
    """
    Test the courses' pre_trained model. Prints the its accuracy on each class, and total accuracy
    """
    print("---------------------- Train Set")
    test_existing_model(BASELINE_MODEL_PATH, TEST_DATA_PATH, to_print=True)
    print("---------------------- Train Set")
    test_existing_model(BASELINE_MODEL_PATH, TRAIN_DATA_PATH, to_print=True)


# execute
#test_existing_model()
