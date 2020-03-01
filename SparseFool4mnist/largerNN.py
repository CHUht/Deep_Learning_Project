import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler

def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.ToTensor(),
                                       ])
    test_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor(),
                                      ])
    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader


trainloader, testloader = load_split_train_test('./data/animals10', .01)
print(trainloader.dataset.classes)

device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")

# model = models.resnet18(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.fc = nn.Sequential(nn.Linear(512, 32),
#                          nn.ReLU(),
#                          nn.Dropout(0.2),
#                          nn.Linear(32, 10),
#                          nn.LogSoftmax(dim=1))
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
# model.to(device)

##################################################################################
# model = models.alexnet(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.classifier[6] = nn.Linear(4096, 10)
# print(model)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.003)
# model.to(device)

##################################################################################
# model = models.squeezenet1_1(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1,1), stride=(1,1))
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.003)
# model.to(device)

##################################################################################
# model = models.vgg11_bn(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False
# num_fits = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_fits, 10)
# print(model)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.003)
# model.to(device)

##################################################################################
model = models.densenet121(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_fits = model.classifier.in_features
model.classifier = nn.Linear(num_fits, 10)
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
model.to(device)


epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        print('training')
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            test_losses.append(test_loss / len(testloader))
            train_losses.append(running_loss / len(trainloader))
            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Test loss: {test_loss / len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy / len(testloader):.3f}")
            running_loss = 0
            model.train()

torch.save(model, 'animals10.pth')
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()