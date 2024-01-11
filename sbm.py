import numpy as np 
import graphlearning as gl 
import plots
import matplotlib.pyplot as plt


def SBM(N0,N1,r,q):

    n = N0+N1
    W1 = np.random.rand(N0,N0) < r
    W2 = np.random.rand(N1,N1) < r
    W12 = np.random.rand(N0,N1) < q
    W = np.zeros((n,n))
    W[:N0,:N0] = W1
    W[N0:,N0:] = W2
    W[:N0,N0:] = W12
    W[N0:,:N0] = W12.T
    np.fill_diagonal(W,0)

    return W

def trial(G,n,N0,beta,p):
    train_ind  = np.random.rand(n) < beta
    train_ind = np.arange(n,dtype=int)[train_ind]
    labels = np.ones(n,dtype=int)
    labels[:N0] = 0
    if p == 2:
        model = gl.ssl.laplace(G.weight_matrix)
        L = model.fit_predict(train_ind,labels[train_ind])
    else:
        u = G.plaplace(train_ind,labels[train_ind],p,tol=1e-3)
        L = (u > 0.5).astype(int)
    return gl.ssl.ssl_accuracy(L, labels, train_ind)

T = 100 #Number of trials (100 takes a long time)
beta = 0.2
r = 0.5

for N in [(1500,1500),(2000,1000)]:
    N0,N1 = N
    n = N0+N1

    if N0 == N1:
        qvals = r/np.linspace(1,2,20)
    else:
        qvals = r/np.linspace(1,10,20)
    plt.figure()
    for p in [2,2.5,3,5]:
        print(N0,N1,p,flush=True)
        acc = np.zeros(len(qvals))
        for i,q in enumerate(qvals):
            for t in range(T):
                W = SBM(N0,N1,r,q)
                G = gl.graph(W)
                if not G.isconnected():
                    print('Graph Not Connected!')
                acc[i] += trial(G,n,N0,beta,p)
            acc[i] /= T
        plt.plot(r/qvals,100-acc,label='$p=%.2f$'%p)
    plt.xlabel('$r/q$')
    plt.ylabel('Error Rate (\%)')
    plt.legend()
    plots.savefig('sbm_%d_%d_%.2f.pdf'%(N0,N1,beta),axis=True,grid=True)



