from segments import *

device = torch.device("cuda:0") # Uncomment this to run on GPU


model = VanillaUNet()

softmax_fn = torch.nn.Softmax2d()
loss_fn2 = torch.nn.CrossEntropyLoss(weight=torch.tensor([.9, .1]))

data = torch.load('checkpoint.30.tar')
model.load_state_dict(data['state_dict'])


train_x, train_y = imageListsToInputData(['unseen/image19.tif'], ['unseen/Mask19.tif'])

batches_x = train_x.unsqueeze(1).float()

train_idxs = np.array(range(train_x.shape[0]))
train_idxs, val_idxs = train_idxs[:-10], train_idxs[-10:]

model = model.to(device)
loss_fn2 = loss_fn2.to(device)
batches_x = batches_x.to(device)
train_y = train_y.to(device)

for i in range(train_x.shape[0]):
    y_pred = model(batches_x[i])
    y_pred = y_pred.to(device)
    
    softmax = softmax_fn(y_pred)
        
    #print(softmax.shape)
    softmaxcpu = softmax.cpu()
    im = Image.fromarray(np.uint8(softmaxcpu[0, 1].data.numpy()*255))
    im.save('output/unseen-i-%d-softmax.png' % i)
    
    #print(softmaxcpu[0,0].min(), softmaxcpu[0,0].max(), softmaxcpu[0,1].min(), softmaxcpu[0,1].max())
    im = Image.fromarray(np.uint8(np.where(softmaxcpu[0, 1].data.numpy() - 0.5 <= 0, 0, 255)))
    im.save('output/unseen-i-%d-softmaxargmax.png' % i)
    
    loss = loss_fn2(y_pred, train_y[i, 1].unsqueeze(0).long())
    print(i, loss.item())
    
    
        
