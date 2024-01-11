import graphlearning as gl 
import matplotlib.pyplot as plt
import plots
import numpy as np 

def trial(G,labels,train_ind,p):
    if p == 2:
        model = gl.ssl.laplace(G.weight_matrix)
        L = model.fit_predict(train_ind,labels[train_ind])
    else:
        u = G.plaplace(train_ind,labels[train_ind],p,tol=1e-3)
        L = (u > 0.5).astype(int)
    return gl.ssl.ssl_accuracy(L, labels, train_ind)

dataset = 'mnist' 
#dataset = 'cifar10' 
classes = (4,9)

if dataset == 'mnist':
    metric = 'vae'
else:
    metric = 'simclr'

#Load data and subset to two classes
data,labels = gl.datasets.load(dataset,metric=metric)
raw_data,labels = gl.datasets.load(dataset,metric='raw')
ind = (labels == classes[0]) | (labels == classes[1])
data = data[ind,:]
raw_data = raw_data[ind,:]
labels = labels[ind]
ind = labels == classes[0]
labels[ind] = 0
labels[~ind] = 1

p = np.random.permutation(raw_data.shape[0])
if dataset == 'mnist':
    img = gl.utils.image_grid(raw_data[p,:],n_rows=5,n_cols=8,return_image=True)
    plots.imsave(dataset+'_sample.png',img,scale=10,gray=True)
else:
    img = gl.utils.color_image_grid(raw_data[p,:],n_rows=5,n_cols=8,return_image=True)
    plots.imsave(dataset+'_sample.png',img,scale=10)

T = 100 #Number of trials (100 takes a long time)
n = data.shape[0]
labels_per_class = [1,2,4,8,16,32,64]
W = gl.weightmatrix.knn(data,10)
G = gl.graph(W)
if not G.isconnected():
    print('Graph Not Connected!')
for p in [2,2.5,3,5]:
    acc = np.zeros(len(labels_per_class))
    for i,r in enumerate(labels_per_class):
        for t in range(T):
            train_ind = gl.trainsets.generate(labels,rate=labels_per_class[i])
            acc[i] += trial(G,labels,train_ind,p)
        acc[i] /= T
    plt.plot(labels_per_class,100-acc,label='$p=%.2f$'%p)
plt.xlabel('Labels per class')
plt.xscale('log')
plt.xticks(labels_per_class,labels_per_class)
plt.ylabel('Error Rate (\%)')
plt.legend()
plots.savefig(dataset+'_plot.pdf',axis=True,grid=True)



