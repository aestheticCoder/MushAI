import os
import cv2
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn 
import torch.nn.functional as F 


# flag to rebuild/construct/train data
REBUILD_DATA = True

class Mush1VsMush2():
    IMG_SIZE = 50
    # directory = /PetImages/Cat/ or /PetImages/Dog
    AmanitaMuscaria = "data/train/AmanitaMuscaria"
    AmanitaPantherina = "data/train/AmanitaPantherina"
    AmanitaPhalloides = "data/train/AmanitaPhalloides"
    AmanitaVirosa = "data/train/AmanitaVirosa"
    Conocybe = "data/train/Conocybe"
    CoprinopsisAtramentaria = "data/train/CoprinopsisAtramentaria"
    CortinariusViolaceus = "data/train/CortinariusViolaceus"
    GalerinaMarginata = "data/train/GalerinaMarginata"
    GyromitraEsculenta = "data/train/GyromitraEsculenta"
    HelvellaVespertina = "data/train/HelvellaVespertina"
    LaetiporusConifericola = "data/train/LaetiporusConifericola"
    PanaeolinaFoenisecii = "data/train/PanaeolinaFoenisecii"
    PaxillusInvolutus = "data/train/PaxillusInvolutus"
    PsilocybeCyanescens = "data/train/PsilocybeCyanescens"
    TurbinellusFloccosus = "data/train/TurbinellusFloccosus"
    
    LABELS = { AmanitaMuscaria: 0, 
               AmanitaPantherina: 1,
               AmanitaPhalloides: 2,
               AmanitaPhalloides: 3,
               Conocybe: 4,
               CoprinopsisAtramentaria: 5,
               CortinariusViolaceus: 6,
               GalerinaMarginata: 7,
               GyromitraEsculenta: 8,
               HelvellaVespertina: 9,
               LaetiporusConifericola: 10,
               PanaeolinaFoenisecii: 11,
               PaxillusInvolutus: 12,
               PsilocybeCyanescens: 13,
               TurbinellusFloccosus: 14
               }

    training_data = []
    # useful for determining "balance"
    amanCount = 0
    cortCount = 0

    # example of np.eye(5)
    # [[1, 0, 0, 0, 0]
    #  [0, 1, 0, 0, 0]
    #  [0, 0, 1, 0, 0]
    #  [0, 0, 0, 0, 1]]
    # example of np.eye(2)
    # [[1, 0]
    #  [0, 1]]
    # np.eye(2)[0] = ([1, 0]) <--- CAT
    # np.eye(2)[1] = ([0, 1]) <-- DOG

    def make_training_data(self):
        # iterate over the Keys AMANITAMUSCARIA/CortinariusViolaceus which are directories
        for label in self.LABELS:
            print(label)
            # iterate over all the images in each directory
            # tqdm is a progress bar to tell how far along you are
            for f in tqdm(os.listdir(label)):
                try:
                    # f is just fileName, we will join that with the label (which is directory)
                    path = os.path.join(label, f)
                    # go to image location, and convert said image to GRAY_SCALE
                    # Grayscale is not required, but color will add an additional "channel"
                    # this added data is not necessary, and so grayscale may be useful for optimization/speed
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # resize image (all inputs must be a standard IMG_SIZE)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    
                    # what is the right type of this img ? 
                    print(label)
                    print(self.LABELS[label])
                    
                    
                    # this will append the image and whether or not it is a cat or dog
                    self.training_data.append([np.array(img), np.eye(15)[self.LABELS[label]]])

                    # determine the balance of dataset (if imbalanced, the NN will learn to always guess that class first and may never overcome/optimize)
                    if label == self.AmanitaMuscaria:
                        self.amanCount += 1
                    elif label == self.CortinariusViolaceus:
                        self.cortCount += 1
                except Exception as e:
                    # if image is no good or corrupt/empty
                    #print(str(e))
                    pass

        # shuffle the training data
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("AmanitaMuscaria: ", self.amanCount)
        print("CortinariusViolaceus: ", self.cortCount)




if REBUILD_DATA:
    amanVsCort = Mush1VsMush2()
    amanVsCort.make_training_data()

# some times you may have to add  " allow_pickle=True "  sometimes not
training_data = np.load("training_data.npy", allow_pickle=True)
# # print(len(training_data))
#print(training_data[14][0])
# # print(training_data[1])

# # print the first item (0) in the second folder (1) which is a dog
# # cmap attempts to color change
# plt.imshow(training_data[1][0], cmap="gray")
# plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # creating a 2d Convolutional Neural Network
        # input is 1, output is 32 features, kernel size is 5x5
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input is 32 (from previous layer) output 64
        self.conv2 = nn.Conv2d(32, 64, 5)
        # input is 54 (from previous layer) output is 128
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        # images from tensor will be a 1 by 50 by 50
        # -1 specifies take input of any size
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(x)

        # fully connected layer input should be calculated as such below
        # cant figure out why it does not add up so we use self._to_linear in lieu
        # formula for output is O = {(W - k + 2*P)/s} + 1 
        # where w = initial image size, k = kernel size, s = stride size, p = padding size
        # input should be (50 - 5 + 2*0)/1  + 1 = 46
        # max pooling where p = pool size/area of Output/p = 46/2 = 23
        # {(23 - 5 + 2*0)/1} + 1 = 19
        # max pooling with floor function o/p = 19/2 = 9
        # {(9-5 + 2*0)/1} + 1 = 5
        # 5/2 = 2
        # i think formula is 2*2*50 or 2*2*128 where 128 is original pixel dimension of image
        
        #self._to_linear is to flatten
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 15)
    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        # print(x[0].shape)
        if self._to_linear is None:
            # x is a "batch of data"
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        # pass through convnet layer
        x = self.convs(x)
        # reshape to be flattened
        x = x.view(-1, self._to_linear)
        # pass through linear regression layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # .softmax() is an activation function
        return F.softmax(x, dim=1)

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
# Mean Squared Error
loss_function = nn.MSELoss()

# reshape just the X values
x = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
# pixel values are between 0 and 255
# by dividing by 255, we will get between 0 and 1
x = x/255.0
y = torch.Tensor([i[1] for i in training_data])

# reserve 10% of data for validation/testing
VAL_PCT = 0.1
val_size = int(len(x)*VAL_PCT)
# print(val_size)

train_x = x[:-val_size]
train_y = y[:-val_size]

test_x = x[-val_size:]
test_y = y[-val_size:]

# print(len(train_x))
# print(len(test_x))

BATCH_SIZE = 200
EPOCHS = 2

for epoch in range(EPOCHS):
    # from 0 to length of size of X, take steps the size of BATCH_SIZE
    for i in tqdm(range(0, len(train_x), BATCH_SIZE)):
        batch_x = train_x[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
        batch_y = train_y[i:i+BATCH_SIZE]
        # set gradient to 0
        net.zero_grad
        outputs = net(batch_x)
        # calculate loss
        loss = loss_function(outputs, batch_y)
        # do backwards propogation on loss
        loss.backward()
        optimizer.step()
print(loss)

correct = 0
total = 0
# no gradient for testing
with torch.no_grad():
    for i in tqdm(range(len(test_x))):
        real_class = torch.argmax(test_y[i])
        net_out = net(test_x[i].view(-1,1,50,50))[0]
        predicted_class = torch.argmax(net_out)
        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy: ", round(correct/total,3))