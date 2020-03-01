# This module
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler


def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
    train_data = datasets.ImageFolder(datadir, transform=train_transforms)
    test_data = datasets.ImageFolder(datadir, transform=test_transforms)
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
device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")


for i in range(4):
	for model_name in ['vgg','alexnet','densenet']:
		print('\n\n\n\n>> training:', 'animals10'+model_name+'_V'+str(i)+'.pth')
		if model_name == 'vgg':
			model = models.vgg11_bn(pretrained=True)
			for param in model.parameters():
				param.requires_grad = False
			num_fits = model.classifier[6].in_features
			model.classifier[6] = nn.Linear(num_fits, 10)
			print(model)

			criterion = torch.nn.CrossEntropyLoss()
			optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.003)
			model.to(device)

		elif model_name == 'alexnet':
			model = models.alexnet(pretrained=True)
			for param in model.parameters():
				param.requires_grad = False
			model.classifier[6] = nn.Linear(4096, 10)
			print(model)

			criterion = torch.nn.CrossEntropyLoss()
			optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.003)
			model.to(device)
		else:
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

		torch.save(model, 'animals10'+model_name+'_V'+str(i)+'.pth')
		print(' >>Model trained saved:', 'animals10'+model_name+'_V'+str(i)+'.pth')