######################################################
# Yasmin Heimann, hyasmin, 311546915
# @output Accuracy decrease rate for different epsilons and 5 perturbated images,
#         with the original image and the moise added.
# @description A module that generates adverarial examples,
#              from a given trained neural network in PtTorch.
######################################################

# IMPORTS #
from __future__ import print_function
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataset import un_normalize_image
from models import SimpleModel
from dataset import get_dataset_as_torch_dataset


# CHANGE THE PATH TO ADJUST YOUR MODEL AND TEST SET #
SAVE_PATH = "./adv_im_examples/"
MODEL_PATH = "./model_checkpoint.ckpt"
TEST_DATA_PATH = "./dev.pickle"

# The list of epsilons to axamine
EPSILONS = [0, .1, .15, .2, .3]
NUM_OF_EXAMPLES = 5


def fgsm_attack(image, epsilon, data_grad):
    """
    The function generates a perturbated image given an image and set of gradients
    :param image: the original image from the test set
    :param epsilon: the number to multiply the gradient by
    :param data_grad: the gradient
    :return: the perturbed image and the noise added to the original image to create the perturbation
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    noise = epsilon*sign_data_grad
    perturbed_image = image + noise
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    noise = torch.clamp(noise, -1, 1)
    # Return the perturbed image
    return perturbed_image, noise


def generate_examples(model, test_loader, epsilon):
    """
    Generates the adversarial examples given a fixed epsilon
    :param model: the model to generate the example from
    :param test_loader: the images data set
    :param epsilon: the fixed number to multiply the gradient by
    :return: the model accuracy with perturbation using the given epsilon and 5 perturbed images with theis noise
    """
    # Accuracy counter
    correct = 0
    adv_examples = []
    model.eval()
    # Loop over all examples in test set
    for data, target in test_loader:
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        loss.backward()
        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data, noise = fgsm_attack(data, epsilon, data_grad)
        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples - the original image
            if (epsilon == 0) and (len(adv_examples) < NUM_OF_EXAMPLES):
                adv_ex = perturbed_data.squeeze()
                adv_noise = noise.squeeze()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex, adv_noise) )
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < NUM_OF_EXAMPLES:
                adv_ex = perturbed_data.squeeze()
                adv_noise = noise.squeeze()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex, adv_noise) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def plot_results(accuracies):
    """
    Plots the accuracies
    :param accuracies: accuracy of the model per each epsilon that was used of perturbing the images
    """
    plt.figure(figsize=(5, 5))
    plt.plot(EPSILONS, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs. Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig("./adv_im_examples/acc_vs_eps")
    # if show is wanted
    # plt.show()


def plot_exapmles(examples):
    """
    Plot several examples of adversarial samples at each epsilon
    :param examples: the examples to plot together
    """
    plt.clf()
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(EPSILONS)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(EPSILONS), len(examples[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(EPSILONS[i]), fontsize=14)
            orig_label, adv_label, adv_exmpl, noise = examples[i][j]
            plt.title("{} -> {}".format(orig_label, adv_label))
            plt.imshow(un_normalize_image(adv_exmpl.detach()), cmap="gray")
    plt.tight_layout()
    plt.savefig(SAVE_PATH + "adv_images")
    # if show is wanted:
    # plt.show()


def save_figs(examples):
    """
    Saves the given examples as images
    :param examples: original, perturbed and noise images
    """
    for i in range(len(EPSILONS)):
        for j in range(len(examples[i])):
            orig_label, adv_label, adv_exmpl, noise = examples[i][j]
            ex_im_name = str(i) + str(j) + "_" +str(orig_label) + "_" + str(adv_label)
            noise_im_name = "noise_" + str(i) + str(j) + "_" + str(orig_label) + "_" + str(adv_label)
            # save the image
            plt.imshow(un_normalize_image(adv_exmpl.detach()), cmap="gray")
            plt.savefig(SAVE_PATH + ex_im_name)
            plt.clf()
            # save the noise
            plt.imshow(un_normalize_image(noise.detach()), cmap="gray")
            plt.savefig(SAVE_PATH + noise_im_name)
            plt.clf()


def generate_adv_exmp():
    """
    The main function that handles the generation of adversarial examples
    It plots and saves 5 adversarial examples and their noise with different epsilons
    """
    # load the model and the data set to manipulate
    model = SimpleModel()
    model.load(MODEL_PATH)
    testset = get_dataset_as_torch_dataset(TEST_DATA_PATH)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,  # todo batch size?
                                              shuffle=False, num_workers=0)
    accuracies = []
    examples = []
    # Generate adversarial examples for each epsilon
    for eps in EPSILONS:
        acc, ex = generate_examples(model, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    # plots results and save the perturbed images
    plot_results(accuracies)
    save_figs(examples)
    plot_exapmles(examples)

# execute
generate_adv_exmp()
