"""
For training the model on the MNIST dataset
output : model(.pth), accuracy plot
"""


import torch
import torchvision
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

from Model import LeNet

def trainmodelandsave(model_name):
    net = LeNet()
    print(net)


    transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    valid = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformImg)

    # create training and validation set indexes (80-20 split)
    idx = list(range(len(train)))
    np.random.seed(1009)
    np.random.shuffle(idx)
    train_idx = idx[ : int(0.8 * len(idx))]
    valid_idx = idx[int(0.8 * len(idx)) : ]

    transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    valid = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformImg)

    # create training and validation set indexes (80-20 split)
    idx = list(range(len(train)))
    np.random.seed(1009)
    np.random.shuffle(idx)
    train_idx = idx[: int(0.8 * len(idx))]
    valid_idx = idx[int(0.8 * len(idx)):]

    # # dataset dimensions
    # print("Training data dimensions: ", train.train_data.shape)
    # print("Test data dimensions: ", test.test_data.shape)
    #
    # # how an image looks in matrix format
    # print("\nAn image in matrix format looks as follows: ", train.train_data[0])

    # generate training and validation set samples
    train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    # Load training and validation data based on above samples
    # Size of an individual batch during training and validation is 30
    # Note that 'SubsetRandomSampler()' function is responsible for providing random samples of data i.e.
    # at every epoch, a batch of 30 records is output and records in every batch are randomly sampled
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, sampler=train_set, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(train, batch_size=32, sampler=valid_set, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test, num_workers=0)

    # set up loss function -- 'SVM Loss' a.k.a ''Cross-Entropy Loss
    loss_func = torch.nn.CrossEntropyLoss()

    # SGD used for optimization, momentum update used as parameter update
    optimization = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Let training begin!
    numEpochs = 5
    training_accuracy = []
    for epoch in range(numEpochs):

        # training set -- perform model training
        epoch_training_loss = 0.0
        num_batches = 0
        for batch_num, training_batch in enumerate(train_loader):  # 'enumerate' is a super helpful function
            # split training data into inputs and labels
            inputs, labels = training_batch  # 'training_batch' is a list
            # wrap data in 'Variable'
            inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
            # Make gradients zero for parameters 'W', 'b'
            optimization.zero_grad()
            # forward, backward pass with parameter update
            forward_output = net(inputs)
            loss = loss_func(forward_output, labels)
            loss.backward()
            optimization.step()
            # calculating loss
            epoch_training_loss += loss.data
            num_batches += 1

        print("epoch: ", epoch, ", loss: ", epoch_training_loss / num_batches)

        # calculate training set accuracy
        accuracy = 0.0
        num_batches = 0
        for batch_num, training_batch in enumerate(train_loader):  # 'enumerate' is a super helpful function
            num_batches += 1
            inputs, actual_val = training_batch
            # perform classification
            predicted_val = net(torch.autograd.Variable(inputs))
            # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function
            predicted_val = predicted_val.data.numpy()
            predicted_val = np.argmax(predicted_val, axis=1)  # retrieved max_values along every row
            # accuracy
            accuracy += accuracy_score(actual_val.numpy(), predicted_val)
        training_accuracy.append(accuracy / num_batches)

    # calculate validation set accuracy
    accuracy = 0.0
    num_batches = 0
    for batch_num, validation_batch in enumerate(valid_loader):  # 'enumerate' is a super helpful function
        num_batches += 1
        inputs, actual_val = validation_batch
        # perform classification
        predicted_val = net(torch.autograd.Variable(inputs))
        # convert 'predicted_val' tensor to numpy array and use 'numpy.argmax()' function
        predicted_val = predicted_val.data.numpy()
        predicted_val = np.argmax(predicted_val, axis=1)  # retrieved max_values along every row
        # accuracy
        accuracy += accuracy_score(actual_val.numpy(), predicted_val)
    print("Validation accuracy :", accuracy / num_batches)

    PATH = f'./Model_trained/{model_name}.pth'
    torch.save(net, PATH)

    epochs = list(range(numEpochs))

    # plotting training and validation accuracies
    fig2 = pyplot.figure()
    pyplot.plot(epochs, training_accuracy, 'r')
    # pyplot.plot(epochs, validation_accuracy, 'g')
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Accuracy")
    pyplot.show()