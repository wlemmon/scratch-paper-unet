# try a few things:
# 1. segment 2d with unets
# 2. segment 2d with capsule nets


# 1. segment 3d with unets
# 2. segment 3d with capsule nets


import numpy as np

import torch


class VanillaUNet(torch.nn.Module):
    def __init__(self):
        super(VanillaUNet, self).__init__()
        
        # there was an attempt to implement padding as in https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/models/unet.py
        # padding 0 is as is in the original paper
        self.padding = 0
        
        #relu = torch.nn.ReLU
        relu = torch.nn.LeakyReLU
        
        # number of conv filters to start with
        #n1 = 64
        n1 = 16
        n2 = n1*2
        n3 = n2*2
        n4 = n3*2
        n5 = n4*2
        
        self.l1_1 = torch.nn.Conv2d(1, n1, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l1_2 = relu()
        self.l1_3 = torch.nn.Conv2d(n1, n1, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l1_4 = relu()
        self.l1_5 = torch.nn.MaxPool2d(2)
        
        self.l2_1 = torch.nn.Conv2d(n1, n2, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l2_2 = relu()
        self.l2_3 = torch.nn.Conv2d(n2, n2, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l2_4 = relu()
        self.l2_5 = torch.nn.MaxPool2d(2)
        
        self.l3_1 = torch.nn.Conv2d(n2, n3, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l3_2 = relu()
        self.l3_3 = torch.nn.Conv2d(n3, n3, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l3_4 = relu()
        self.l3_5 = torch.nn.MaxPool2d(2)
        
        self.l4_1 = torch.nn.Conv2d(n3, n4, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l4_2 = relu()
        self.l4_3 = torch.nn.Conv2d(n4, n4, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l4_4 = relu()
        self.l4_5 = torch.nn.MaxPool2d(2)
        
        self.l5_1 = torch.nn.Conv2d(n4, n5, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l5_2 = relu()
        self.l5_3 = torch.nn.Conv2d(n5, n5, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l5_4 = relu()
        
        self.l6_1 = torch.nn.ConvTranspose2d(n5, n4, kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.l6_2 = torch.nn.Conv2d(n5, n4, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l6_3 = relu()
        self.l6_4 = torch.nn.Conv2d(n4, n4, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l6_5 = relu()
        
        self.l7_1 = torch.nn.ConvTranspose2d(n4, n3, kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.l7_2 = torch.nn.Conv2d(n4, n3, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l7_3 = relu()
        self.l7_4 = torch.nn.Conv2d(n3, n3, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l7_5 = relu()
        
        self.l8_1 = torch.nn.ConvTranspose2d(n3, n2, kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.l8_2 = torch.nn.Conv2d(n3, n2, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l8_3 = relu()
        self.l8_4 = torch.nn.Conv2d(n2, n2, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l8_5 = relu()
        
        self.l9_1 = torch.nn.ConvTranspose2d(n2, n1, kernel_size=2, stride=2, padding=0, output_padding=0, dilation=1, groups=1, bias=True)
        self.l9_2 = torch.nn.Conv2d(n2, n1, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l9_3 = relu()
        self.l9_4 = torch.nn.Conv2d(n1, n1, 3, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        self.l9_5 = relu()
        
        self.l10_1 = torch.nn.Conv2d(n1, 2, 1, stride=1, padding=self.padding, dilation=1, groups=1, bias=True)
        
        
        torch.nn.init.xavier_normal_(self.l1_1.weight)
        torch.nn.init.xavier_normal_(self.l1_3.weight)
        
        torch.nn.init.xavier_normal_(self.l2_1.weight)
        torch.nn.init.xavier_normal_(self.l2_3.weight)
        
        torch.nn.init.xavier_normal_(self.l3_1.weight)
        torch.nn.init.xavier_normal_(self.l3_3.weight)
        
        torch.nn.init.xavier_normal_(self.l4_1.weight)
        torch.nn.init.xavier_normal_(self.l4_3.weight)
        
        torch.nn.init.xavier_normal_(self.l5_1.weight)
        torch.nn.init.xavier_normal_(self.l5_3.weight)
        
        torch.nn.init.xavier_normal_(self.l6_1.weight)
        torch.nn.init.xavier_normal_(self.l6_2.weight)
        torch.nn.init.xavier_normal_(self.l6_4.weight)
        
        torch.nn.init.xavier_normal_(self.l7_1.weight)
        torch.nn.init.xavier_normal_(self.l7_2.weight)
        torch.nn.init.xavier_normal_(self.l7_4.weight)
        
        torch.nn.init.xavier_normal_(self.l8_1.weight)
        torch.nn.init.xavier_normal_(self.l8_2.weight)
        torch.nn.init.xavier_normal_(self.l8_4.weight)
        
        torch.nn.init.xavier_normal_(self.l9_1.weight)
        torch.nn.init.xavier_normal_(self.l9_2.weight)
        torch.nn.init.xavier_normal_(self.l9_4.weight)
        
        torch.nn.init.xavier_normal_(self.l10_1.weight)
        
    def forward(self, x):
    
        
        #print (x.shape)
        x1_1 = self.l1_1(x)
        #print (x1_1.shape)
        x1_2 = self.l1_2(x1_1)
        #print (x1_2.shape)
        x1_3 = self.l1_3(x1_2)
        #print (x1_3.shape)
        x1_4 = self.l1_4(x1_3)
        #print (x1_4.shape)
        x1_5 = self.l1_5(x1_4)
        #print (x1_5.shape)
        
        del x1_1
        del x1_2
        del x1_3
        
        
        x2_1 = self.l2_1(x1_5)
        #print (x2_1.shape)
        x2_2 = self.l2_2(x2_1)
        #print (x2_2.shape)
        x2_3 = self.l2_3(x2_2)
        #print (x2_3.shape)
        x2_4 = self.l2_4(x2_3)
        #print (x2_4.shape)
        x2_5 = self.l2_5(x2_4)
        #print (x2_5.shape)
        #print()
        
        del x2_1
        del x2_2
        del x2_3
        
        
        x3_1 = self.l3_1(x2_5)
        x3_2 = self.l3_2(x3_1)
        x3_3 = self.l3_3(x3_2)
        x3_4 = self.l3_4(x3_3)
        x3_5 = self.l3_5(x3_4)
        
        del x3_1
        del x3_2
        del x3_3
        
        x4_1 = self.l4_1(x3_5)
        x4_2 = self.l4_2(x4_1)
        x4_3 = self.l4_3(x4_2)
        x4_4 = self.l4_4(x4_3)
        x4_5 = self.l4_5(x4_4)
        
        del x4_1
        del x4_2
        del x4_3
        
        x5_1 = self.l5_1(x4_5)
        x5_2 = self.l5_2(x5_1)
        x5_3 = self.l5_3(x5_2)
        x5_4 = self.l5_4(x5_3)
        #print(x5_4.shape)
        
        del x5_1
        del x5_2
        del x5_3
        
        
        
        x6_1 = self.l6_1(x5_4)
        #print(x6_1.shape)
        crop = (x4_4.shape[-1] - x6_1.shape[-1])//2
        #print (x4_4[:,:,crop:-crop,crop:-crop].shape, x6_1.shape)
        #print(x6_1.shape)
        if self.padding == 1:
            x6_1 = torch.cat([x4_4,x6_1], dim=1)
        elif self.padding == 0:
            x6_1 = torch.cat([x4_4[:,:,crop:-crop,crop:-crop],x6_1], dim=1)
        #print(x6_1.shape)
        #sdf
        x6_2 = self.l6_2(x6_1)
        #print(x6_2.shape)
        x6_3 = self.l6_3(x6_2)
        #print(x6_3.shape)
        x6_4 = self.l6_4(x6_3)
        #print(x6_4.shape)
        x6_5 = self.l6_5(x6_4)
        #print(x6_5.shape)
        
        del x5_4
        del x4_4
        del x6_1
        del x6_2
        del x6_3
        del x6_4
        

        x7_1 = self.l7_1(x6_5)
        #print(x7_1.shape)
        crop = (x3_4.shape[-1] - x7_1.shape[-1])//2
        if self.padding == 1:
            x7_1 = torch.cat([x3_4,x7_1], dim=1)
        elif self.padding == 0:
            x7_1 = torch.cat([x3_4[:,:,crop:-crop,crop:-crop],x7_1], dim=1)
        #print(x7_1.shape)
        x7_2 = self.l7_2(x7_1)
        #print(x7_2.shape)
        x7_3 = self.l7_3(x7_2)
        #print(x7_3.shape)
        x7_4 = self.l7_4(x7_3)
        #print(x7_4.shape)
        x7_5 = self.l7_5(x7_4)
        #print(x7_5.shape)
        
        del x6_5
        del x3_4
        del x7_1
        del x7_2
        del x7_3
        del x7_4
        
        x8_1 = self.l8_1(x7_5)
        crop = (x2_4.shape[-1] - x8_1.shape[-1])//2
        x8_1 = torch.cat([x2_4[:,:,crop:-crop,crop:-crop],x8_1], dim=1)
        x8_2 = self.l8_2(x8_1)
        x8_3 = self.l8_3(x8_2)
        x8_4 = self.l8_4(x8_3)
        x8_5 = self.l8_5(x8_4)
        
        del x7_5
        del x2_4
        del x8_1
        del x8_2
        del x8_3
        del x8_4
        
        x9_1 = self.l9_1(x8_5)
        crop = (x1_4.shape[-1] - x9_1.shape[-1])//2
        x9_1 = torch.cat([x1_4[:,:,crop:-crop,crop:-crop],x9_1], dim=1)
        x9_2 = self.l9_2(x9_1)
        x9_3 = self.l9_3(x9_2)
        x9_4 = self.l9_4(x9_3)
        x9_5 = self.l9_5(x9_4)
        
        del x8_5
        del x1_4
        del x9_1
        del x9_2
        del x9_3
        del x9_4
        
        #print(x9_5.shape)
        x10_1 = self.l10_1(x9_5)
        #print(x10_1.shape)
        #print ('forward done')
        return x10_1


from numpy.random import rand, shuffle

from keras.preprocessing.image import *
        
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates

from PIL import Image, ImageSequence
from torchvision import models, transforms
from torch.autograd import Variable

# Function to distort image
def elastic_transform(image, alpha=2000, sigma=40, alpha_affine=40, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[1:3]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    for i in range(shape[0]):
        image[i,:,:] = cv2.warpAffine(image[i,:,:], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    image = image.reshape(shape)

    blur_size = int(4*sigma) | 1

    dx = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape_size) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[2]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    def_img = np.zeros_like(image)
    for i in range(shape[0]):
        def_img[i,:,:] = map_coordinates(image[i,:,:], indices, order=1).reshape(shape_size)

    return def_img


def salt_pepper_noise(image, salt=0.2, amount=0.004):
    chan, row, col = image.shape
    num_salt = np.ceil(amount * row * salt)
    num_pepper = np.ceil(amount * row * (1.0 - salt))

    
    for n in range(chan//2): # //2 so we don't augment the mask
        # Add Salt noise
        #print (image.shape[1:3])
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[1:3]]
        #print(coords)
        #print ("HI")
        image[n, coords[0], coords[1]] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[1:3]]
        image[n, coords[0], coords[1]] = 0

    return image

    
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def augmentImages(batch_of_images, batch_of_masks):
    for i in range(len(batch_of_images)):
        #print(batch_of_images.shape)
        img_and_mask = np.concatenate((batch_of_images[i, ...], batch_of_masks[i,...]), axis=0)
        if img_and_mask.ndim == 4: # This assumes single channel data. For multi-channel you'll need
            # change this to put all channel in slices channel
            orig_shape = img_and_mask.shape
            img_and_mask = img_and_mask.reshape((img_and_mask.shape[0:3]))

        #if np.random.randint(0,10) == 7:
        #    img_and_mask = random_rotation(img_and_mask, rg=45, row_axis=1, col_axis=2, channel_axis=0,
        #                                   fill_mode='constant', cval=0.)

        #if np.random.randint(0, 5) == 3:
        img_and_mask = elastic_transform(img_and_mask, alpha=1000, sigma=80, alpha_affine=50)

        #if np.random.randint(0, 10) == 7:
        #img_and_mask = random_shift(img_and_mask, wrg=0.2, hrg=0.2, row_axis=1, col_axis=2, channel_axis=0,
        #                                fill_mode='constant', cval=0.)

        #if np.random.randint(0, 10) == 7:
        #    img_and_mask = random_shear(img_and_mask, intensity=16, row_axis=1, col_axis=2, channel_axis=0,
        #                 fill_mode='constant', cval=0.)

        #if np.random.randint(0, 10) == 7:
        #    img_and_mask = random_zoom(img_and_mask, zoom_range=(0.75, 0.75), row_axis=1, col_axis=2, channel_axis=0,
        #                 fill_mode='constant', cval=0.)

        #if np.random.randint(0, 10) == 7:
        if np.random.randint(0, 2) == 1:
            img_and_mask = flip_axis(img_and_mask, axis=1)

        #if np.random.randint(0, 2) == 7:
        if np.random.randint(0, 2) == 1:
            img_and_mask = flip_axis(img_and_mask, axis=2)

        #if np.random.randint(0, 10) == 7:
        #    salt_pepper_noise(img_and_mask, salt=0.2, amount=0.04)

        if batch_of_images.ndim == 4:
            batch_of_images[i, ...] = img_and_mask[0:img_and_mask.shape[0]//2,...]
            batch_of_masks[i,...] = img_and_mask[img_and_mask.shape[0]//2:,...]
        if batch_of_images.ndim == 5:
            img_and_mask = img_and_mask.reshape(orig_shape)
            batch_of_images[i, ...] = img_and_mask[...,0:img_and_mask.shape[0]//2, :]
            batch_of_masks[i,...] = img_and_mask[...,img_and_mask.shape[0]//2:, :]

        # Ensure the masks did not get any non-binary values.
        batch_of_masks[batch_of_masks > 0.5] = 1
        batch_of_masks[batch_of_masks <= 0.5] = 0

    return(batch_of_images, batch_of_masks)

  
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
  
preprocess_x = transforms.Compose([
   transforms.ToPILImage(),
   #transforms.Scale(572),
   #transforms.CenterCrop(768),
   transforms.CenterCrop(572),
   #transforms.CenterCrop(300),
   transforms.ToTensor(),
   #normalize
])

preprocess_y = transforms.Compose([
   transforms.ToPILImage(),
   #transforms.Scale(572),
   #transforms.CenterCrop(768),
   transforms.CenterCrop(388),
   #transforms.CenterCrop(116),
   transforms.ToTensor(),
   #normalize
])

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
        

def imageListToInputData(imageList):
    train_x = []
    for imageName in imageList:
        with Image.open(imageName) as img:
            train_x.append(np.array(img, dtype='float')/255.)
    #train_x = [x.data.numpy() for x in train_x]
    train_x = np.stack(train_x, axis=0)
    train_x = np.expand_dims(train_x, axis=1)

    return train_x
def imageListsToInputData(xList, yList):


    train_x = imageListToInputData(xList)
    train_y = imageListToInputData(yList)

    all_train_x = [train_x]
    all_train_y = [train_y]


    for i in range(5):
        train_x2, train_y2 = augmentImages(train_x.copy(), train_y.copy())
        all_train_x.append(train_x2)
        all_train_y.append(train_y2)
        
    train_x = np.concatenate(all_train_x, axis=0)
    train_y = np.concatenate(all_train_y, axis=0)


    for i in range(train_x.shape[0]):
        im = Image.fromarray(train_x[i, 0]*255).convert("L")
        im.save('output/train_%d.png' % i)
        
    for i in range(train_y.shape[0]):
        #x = train_y[i, 0]*255
        im = Image.fromarray(train_y[i, 0]*255).convert("L")
        im.save('output/train_gt%d.png' % i)

        
    print(np.moveaxis(train_x[0], 0, 2).shape)

    train_x = (np.moveaxis(train_x, 1, 3)*255).astype('uint8')
    train_y = (np.moveaxis(train_y, 1, 3)*255).astype('uint8')


    train_x = np.stack([preprocess_x(train_x[i]) for i in range(train_x.shape[0])])
    train_y = np.stack([preprocess_y(train_y[i]) for i in range(train_y.shape[0])])


    print (train_x.min(), train_x.max())
    print (train_y.min(), train_y.max())

    train_y = np.concatenate([train_y == 0, train_y == 1], axis=1).astype('float')

    print (train_y.shape)

    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).float()

    print('here2', train_x.shape, train_y.shape)

    for i in range(train_y.shape[0]):
        #im = Image.fromarray(np.uint8(train_y[i, 0].data.numpy()*255))
        #im.save('output/train%d-class-0.png' % i)
        #im = Image.fromarray(np.uint8(train_y[i, 1].data.numpy()*255))
        #im.save('output/train%d-class-1.png' % i)
        im = Image.fromarray(np.uint8(train_y[i, 1].data.numpy()*255))
        im.save('output/train%d.png' % i)
        


    #train_y2 = train_y.long()
    #train_y2 = train_y2[:,1,:,:]
    return train_x, train_y


def train():
    
    #trainidx = [0, 40]#, 60, 80, 120, 100, 140, 160]
    trainidx = range(160)[::4]

    #device = torch.device("cpu")
    device = torch.device("cuda:0") # Uncomment this to run on GPU

    model = VanillaUNet()

    softmax_fn = torch.nn.Softmax2d()
    loss_fn = torch.nn.MSELoss()
    loss_fn2 = torch.nn.CrossEntropyLoss(weight=torch.tensor([.9, .1]))
    #loss_fn2 = torch.nn.CrossEntropyLoss(weight=None)
    learning_rate = 1e-4



    train_x, train_y = imageListsToInputData(["training%d.png" % i for i in trainidx], ["training_ground_truth%d.png" % i for i in trainidx])

    #optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # batches of size 1. is this okay to do?
    batches_x = train_x.unsqueeze(1).float()

    # use the no channel version of training data for cross entropy loss
    #batches_y_3dims = train_y[:,1,:,:]


    train_idxs = np.array(range(train_x.shape[0]))
    train_idxs, val_idxs = train_idxs[:-10], train_idxs[-10:]

    model = model.to(device)
    loss_fn2 = loss_fn2.to(device)
    #model.cuda()
    #train_y572.to(device)
    batches_x = batches_x.to(device)
    train_y = train_y.to(device)

    for t in range(500):
        
        np.random.shuffle(train_idxs)
        for i in train_idxs:
            optimizer.zero_grad()
            
            
            #batch_y2 = train_y2[i].unsqueeze(0)
            
            #print(batch_x.shape)
            #print ('hi')
            #y_pred = model(train_y572)
            y_pred = model(batches_x[i])
            y_pred = y_pred.to(device)
            #print(type(y_pred))
            #print(y_pred.data.numpy() - batch_y.data.numpy())
            #print(((y_pred.data.numpy() - batch_y.data.numpy())**2).sum())
            
            #x = y_pred[0, 0].data.numpy().copy()
            #x += x.min()
            #x /= x.max()
            #im = Image.fromarray(np.uint8(x*255))
            #im.save('output/out-i-%d-t-%d-class-0.png' % (i, t))
            #
            #x = y_pred[0, 1].data.numpy().copy()
            #x += x.min()
            #x /= x.max()
            #im = Image.fromarray(np.uint8(x*255))
            #im.save('output/out-i-%d-t-%d-class-1.png' %  (i, t))
            
            #print(train_y.shape, y_pred.shape)
            if t % 10 == 0:
                softmax = softmax_fn(y_pred)
                
                #print(softmax.shape)
                softmaxcpu = softmax.cpu()
                im = Image.fromarray(np.uint8(softmaxcpu[0, 1].data.numpy()*255))
                im.save('output/out-i-%d-t-%d-softmax.png' % (i, t))
                
                #print(softmaxcpu[0,0].min(), softmaxcpu[0,0].max(), softmaxcpu[0,1].min(), softmaxcpu[0,1].max())
                im = Image.fromarray(np.uint8(np.where(softmaxcpu[0, 1].data.numpy() - 0.5 <= 0, 0, 255)))
                im.save('output/out-i-%d-t-%d-softmaxargmax.png' % (i, t))
                
                save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, False, filename='checkpoint.%d.tar' % t)
            
            
            #loss = loss_fn(softmax[0, 1], train_y[0, 1])
            
            #print(y_pred.shape, train_y[i, 1].unsqueeze(0).long().shape)
            #print(y_pred.type(), train_y[i, 1].type, train_y[i, 1].unsqueeze(0).long().type())
            loss = loss_fn2(y_pred, train_y[i, 1].unsqueeze(0).long())
            
            #print(loss.shape)
            print(t, loss.item())
            
            
            loss.backward()
            
            optimizer.step()
        if t % 10 == 0:
            for i in val_idxs:
                
                y_pred = model(batches_x[i])
                
                softmax = softmax_fn(y_pred)
                
                softmaxcpu = softmax.cpu()
                im = Image.fromarray(np.uint8(softmaxcpu[0, 1].data.numpy()*255))
                im.save('output/val-i-%d-t-%d-softmax.png' % (i, t))
                
                im = Image.fromarray(np.uint8(np.where(softmaxcpu[0, 1].data.numpy() - 0.5 <= 0, 0, 255)))
                im.save('output/val-i-%d-t-%d-softmaxargmax.png' % (i, t))
            
if __name__ == '__main__':
    train()
