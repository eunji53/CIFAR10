import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import numpy as np
from tqdm import tqdm

model_num = 5 # total number of models
total_epoch = 50 # total epoch
lr = 0.01 # initial learning rate

for s in range(model_num):
    # fix random seed
    seed_number = s
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define the data transforms
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = model.to(device)  # Move the model to the GPU

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    def train():
        model.train()
        running_loss = 0.0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0   
                
    def test():
        model.eval()
        
        # Test the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validationloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %f %%' % (100 * correct / total))

    # 데이터셋을 Train과 나머지로 분리합니다.
    indices = list(range(len(dataset)))
    random.shuffle(indices)  # 인덱스를 섞습니다.

    for i in range(5): # 5-fold cross validation
        # 데이터셋을 trainset 40000개와 validationset 10000개로 분리합니다.
        train_indices = indices[:40000-i*10000] + indices[50000-i*10000:]
        validation_indices = indices[40000-i*10000:50000-i*10000]

        trainset =  torch.utils.data.Subset(dataset, train_indices)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16)
        validationset =  torch.utils.data.Subset(dataset, validation_indices)
        validationloader = torch.utils.data.DataLoader(validationset, batch_size=100, shuffle=False, num_workers=16)


        # Train the model
        for epoch in tqdm(range(total_epoch)):
            train()
            test()
            scheduler.step()

        print('Finished Training')

        # Save the checkpoint of the last model
        PATH = './resnet18_cifar10_%dfold_%f_%d.pth' % (i+1, lr, seed_number)
        torch.save(model.state_dict(), PATH)
