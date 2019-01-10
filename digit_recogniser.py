"""# Reconising Handwritten-Digits

In this notebook I build a simple neural network with multiple layer perceptron that will take 
an image of a handwritten digit and predict what exactly is the number.
I use MNIST dataset which contains 60,000 images of handwritten-digits for training and 10,000 images for testing. The images
are grayscale and of size 28x28 pixels.

I will use PyTorch  to build the model. 
It is a new framework for deep learning that was introduced in the early 2017.
However, even though PyTorch is quite new in compare to TensorFlow, Caffe2 or CNTK,  it has already a pretty big impact in the deep learning community.

## Data Loading
In the following, I first import PyTorch and load the MNIST dataset which is available in the `torchvision` package. I  then load the dataset using `Dataloader`, a nice module to load  dataset available in PyTorch.
"""

import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

len(trainloader.dataset)

len(testloader.dataset)

"""## Data Viewing 
Now let us look at the data. It is loaded into `trainloader` and `testloader`where  the batch size is 64 for the training  and 1000 for testing.
 The batch size basically is the number of images I get in one iteration from the data loader.
I set the shuffle mode for the training data to be true and for the testing data to be false.
Basically, by setting the shuffle mode to true, the 
 dataset will get shuffled   every time I start going through the data loader again. 
 
Now I grab the first batch in the testloader to check out the data. 
I first make an iterator with `iter(testloader)` which I will loop later for testing. 
As it is shown below, the first batch is just a tensor with size (1000, 1, 28, 28).  It means we have 1000  images in the batch where each image has 1 color channel (since it is a grayscale) and the size of each image is 28x28 pixels.
"""

dataiter = iter(testloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

"""By using matplotlib, I can  show how the images look like.   I plot the first 6 images from the first batch together withs their label."""

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(images[i][0], cmap='gray', interpolation='none')
  plt.title("{}".format(labels[i]))
  plt.xticks([])
  plt.yticks([])

"""## Building the Model

I will build a model by using a simple neural network with two (fully-connected)  hidden layers.
Beside the hidden layers, the model also has an input and output layers.
I will make the model to have 784 units in the input layer. This is because we will consider a grayscale image with 28x28 pixels as an input.
For the output layer I consider 10 units. This is because I want the model to predict the digit shown in the input image, i.e.
whether it is $0, 1, \ldots, 9$.
The model will calculate probabilities of the image being in each of these classes. For the hidden layers, I set the first one to have 128 units
and 64 units for the second one.

For the activation function I choose rectified linear units (or ReLU in short) and I will not use any dropout layer. 

In PyTorch it is very handy to build a neural network. We can use `nn`
 module  that provides an efficient way to build large neural networks. 
 It also automatically initializes the weights and biases of the network.

Since the tensor will be passed sequentially through the layers, I  use `nn.Sequential` as follows.
"""

from torch import nn
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

"""So I have build the model I want. The model will output  the log-probabilities of the input images being $0, 1, \ldots, 9$. 
Now let us see what happens if I pass an image to this model.
I will take the first image on the first batch I have considered before. However, since I have created a model that 
takes as an input image, a one dimension tensor, i.e.
1 x 784, I need todo some preprocessing. This is because the images I have considered before are still two dimension tensors of size 28 x 28.
"""

images.resize_(64, 1, 784) # resize images into a one dimension vector  
output = model.forward(images[0,:]) # output of the network 
ps = torch.exp(output)  # get the probability of the image being 0,1,...,9

ps

"""To see this more clearly,  I create a function to display the image and probablities."""

import matplotlib.pyplot as plt
import numpy as np

def view_probs(img, ps):
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='gray')
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    
    plt.tight_layout()


img = images[0]
view_probs(img.view(1, 28, 28), ps)

"""As it is shown, the model was all over the classes with its predictions.
It has no idea what this digit is. 
This is because I have not  trained the model yet, so all the weights are random!

## Training the Model

Now let us train the model. First I initialize the criterion and the optimizer that I will use for the training.
 I will train the network for 5 epochs. In other words,  it will see  the training dataset five times.
"""

criterion = nn.NLLLoss()

from torch import optim
optimizer = optim.SGD(model.parameters(), lr=0.003)

n_epochs = 5

# create list for saving training and testing loss
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(trainloader.dataset) for i in range(n_epochs + 1)]

"""I check the initial testing loss before the model is being trained, and then train for 5 epochs."""

valid_loss = 0
correct = 0
model.eval()
for data, target in testloader:
  # forward pass: compute predicted outputs by passing inputs to the model
  data= data.view(data.shape[0], -1)
  out = model(data)
  # calculate the batch loss
  vloss = criterion(out, target)
  # update average validation loss 
  valid_loss += vloss.item()*data.size(0)
  pred = out.data.max(1, keepdim=True)[1]
  correct += pred.eq(target.data.view_as(pred)).sum()
else:
  test_losses.append(valid_loss/len(testloader.dataset))
  print(f"Test loss: {valid_loss/len(testloader.dataset)}")
  print('Accuracy: {}/{} ({:.0f}%)\n'.format(
    correct, len(testloader.dataset),
    100. * correct / len(testloader.dataset)))

for e in range(1, n_epochs+1):
    batch_idx = 0
    running_loss = 0
    valid_loss = 0
    log_interval = 10
    correct = 10
    
# TRAINING THE MODEL
    model.train()
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
        
        batch_idx += 1 
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % log_interval == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(e, batch_idx * len(images), len(trainloader.dataset), 100. * batch_idx / len(trainloader), loss.item()))
           train_losses.append(loss.item())
           train_counter.append((batch_idx*64) + ((e-1)*len(trainloader.dataset)))
    else:
        print(f"Training loss: {running_loss/len(trainloader)}") 

# VALIDATING THE MODEL       
    model.eval()
    for data, target in testloader: 
        # forward pass: compute predicted outputs by passing inputs to the model
        data= data.view(data.shape[0], -1)
        out = model(data)
        # calculate the batch loss
        vloss = criterion(out, target)
        # update average validation loss 
        valid_loss += vloss.item()*data.size(0)
        # keep track accuracy 
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    else:
        test_losses.append(valid_loss/len(testloader.dataset))
        print(f"Test loss: {valid_loss/len(testloader.dataset)}")
        print('Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))

"""Now the model has been trained. It can recognise the handwritten digit without any doubt."""

img = img.view(1, 784) 
 
ps_img = torch.exp(model(img))
view_probs(img.view(1, 28, 28), ps_img)

"""## Model Evaluation

In the beginning, as expected we only get about 10% accuracy. However  after training with  just 5 epochs  by using a very simple neural network we already managed to achieve 89% accuracy. 

Here I plot the training curve.  By looking at it, it seems that training the model for 5 epochs is the best thing I can do.  This is because continue training for   more epochs probably will make the model deals with overfitting issues.
"""

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss','Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
