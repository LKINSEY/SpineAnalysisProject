#%%
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import datetime
start_time = datetime.datetime.now()
#%%
exampleMov = 'path/to/your/tiff/stack' #assumes stack is motion-corrected
stack = tiff.imread(exampleMov)
#%%
print('Loaded Stack Shape: \n', stack.shape)
reshapedStack = np.reshape(stack, ( stack.shape[0], stack.shape[1] * stack.shape[2])).T
plt.imshow(reshapedStack)
plt.title(f'{stack.shape} reshaped to {reshapedStack.shape}')
plt.xlabel('Time (frame #)')
plt.ylabel('Pixel ID')
print('Reshaped Size: \n', reshapedStack.shape) #first index is the time, second index is the pixel value
# %%
#Functions
def find_closest_centroids(X,centroids):
    '''
    Computes the centroid memberships for each example

    Inputs:
        X == data (m,n)
        centroids == centroids (K,n) #n is a vector of features
    Outputs:
        idx == (m,) closest centroids ## the "c_i" in the math equations, i.e. a centroid identity for each datapoint
    '''
    #Setting k
    K = centroids.shape[0]
    #initializing return
    idx = np.zeros(X.shape[0], dtype = int)
    for mi in range(X.shape[0]):
        #for each centroid
        distVec = np.zeros(K)
        for c in range(K):
            distVec[c] = sum((X[mi] - centroids[c])**2) #calculate distance of data point mi from center of centroid c
        idx[mi] = np.argmin(distVec)
    return idx
def update_centroids(X,idx,K):
    '''
    Returns the new centroids by computing the means of the data points assigned to each centroid
    Inputs:
        X ==> data (m,n)
        idx ==> array containing index assignments of each data point to a centroid (m,)
        K ==> number of centroids (int)
    Outputs:
        centroids ==> an array with updated centroid coordinates (K,n)
    '''
    m,n = X.shape
    centroids = np.zeros((K,n))
    #need to clean data so it does not throw me -infs
    X = np.where(np.isinf(X), np.nan, X)

    #for each centroid
    for k in range(K): 
        cluster = X[idx==k] #grabbing coords that belong to centroid k
        centroids[k] = np.nanmean(cluster, axis=0)
        print(centroids)
    return centroids
def run_kMeans(X, initial_centroids, max_iters=10):
    '''
    runs the k means algorithm on data matix X, where each row of X is a single example
    '''
    m,n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m) #initialize storage space
    for i in range(max_iters):
        #output progress 
        print("K-Means iteration %d/%d" % (i, max_iters-1))
        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)        
        # Given the memberships, compute new centroids
        centroids = update_centroids(X, idx, K)
        print(centroids)
    return centroids, idx
def kmeans_init_centroids(X,K):
    '''
    This function initializes K centroids that are to be 
    used in k means on the dataset x
    Inputs:
        X = data poiunts
        K = number of centroids/clusters
    Outputs
        centroids: initial coords of initial centroids
    '''
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    count = 0
    while np.any(np.isnan(centroids)): #we make sure there are no nans in our initialized centroids
        randidx = np.random.permutation(X.shape[0])
        centroids = X[randidx[:K]] #take the first K examples as centroids
        count +=1
        if count > 1e3:
            print('Cant initialize correctly...')
            return
    return centroids
#%%
def create_img_features(tiffMov):
    '''
    takes the reshaped img and creates 2 transforms, 1 is avg zscore and 2 is variance.
    Takes an (x*y , t) array and reduces it to a (x*y, 2) array where 2 is the number of features created
    Inputs:
        tiffMov ==> reshaped tiff movie reshaped to (x*y, t) each col is a frame
    Outputs:
        X_data ==> a feature array to be used for clustering where size is (x*y, 2)
            2 == n
            when n==1: avg of z score
            when n==2: variance of z score
    '''
    avgPix = np.nanmean(tiffMov, axis=1)
    avgPix = np.reshape(avgPix, (len(avgPix),1))
    stdPix = np.nanstd(tiffMov, axis=1)
    stdPix = np.reshape(stdPix, (len(stdPix),1))
    regularizedMov = (tiffMov - avgPix) / stdPix
    feature1 = (np.nanmax(tiffMov, axis=1)) ##edited, taking the max makes more sense, if the sensor has really good snr
    feature1 = np.reshape(feature1, (len(feature1),1))

    #what matters most is this feature right here... #modifying again to make it skew instead of var
    feature2 =(np.nanmean(tiffMov, axis=1)) ##### changed changed again by log transforming because numbers so big np.sum bugs out
    feature2 = np.reshape(feature2, (len(feature2),1))

    #unless we add more features?? (but then we won't be able to plot it on a 2d scatter..)
    feature3 = np.log(np.nanvar(regularizedMov, axis=1))
    feature3 = np.reshape(feature3, (len(feature3),1))

    feature4 = np.log(np.nanstd(regularizedMov, axis=1))
    feature4 = np.reshape(feature4, (len(feature4),1))

    print('Feature 1 shape: ', feature1.shape)
    print('Feature 2 shape: ', feature2.shape)
    print('We have so many features now....')
    return np.concatenate((feature1, feature2, feature3, feature4), axis=1)

Xdata = create_img_features(reshapedStack).T
#we just want to visualize clusters, ideally background, stucture, and synapse pixels
#note that we can kind of see motion effects when we imshow(reshapedStack) -- maybe just throw away these bullshit frames?
#we can do that later, do it by making a distribution of the location of the "not non pixels" in each col, then if the avg of that dist is beyond 1 sigma throw that shit away
print('Feature Array Shape is: \n', Xdata.shape)

K = 4#seems like some points on dendrite are actually showing up as cluster 1.... wonder why... cluster 1 is background, cluster 2 is noise, cluster 3 is structural, can we make cluster 4 synapses?
max_iters = 20
#Running our k means alg on Xdata
initial_centroids = kmeans_init_centroids(Xdata.T, K)
print(initial_centroids)
centroids, idx = run_kMeans(Xdata.T, initial_centroids, max_iters)
print(centroids)
for feature in range(K):
    summaryImage = idx.reshape((stack.shape[1], stack.shape[2]))==feature
    plt.figure(feature)
    plt.imshow(summaryImage)
    plt.title(f'Summary Image {feature}')
#%%
clusterToSee = 1
datax = Xdata[0]
datay = Xdata[2]
plt.figure(1)
colors = ['green', 'blue', 'orange', 'pink', 'red']
legendKey = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'cluster 4', 'cluster 5'] #'Cluster Centers'] #'Cluster 1', 'Cluster 1']
fig, ax = plt.subplots(1,K-1, figsize=(12,4))

feature1 =  Xdata[0]
for feature in range(Xdata.shape[0]-1):   
    feature2 =  Xdata[feature+1]
    for clusterNum in range(K):
        xx = feature1[idx == clusterNum]
        yy = feature2[idx == clusterNum]
        ax[feature].scatter(xx,yy, color = colors[clusterNum])
        ax[feature].set_xlabel(f'Feature {feature+1}')


# fig.text(0.5, 0.04,'Feature', ha='center', va='center', fontsize=14)#xlabel
fig.text(0.04, 0.5,'Feature 0', ha='center', va='center', rotation='vertical', fontsize=14) #ylabel
fig.suptitle('Plotting Features Against Each Other to Look for Relationship')
plt.show()
#%%

end_time = datetime.datetime.now()
duration = end_time - start_time
print(f'Total Time To Extract: {duration}')
#%%

#As long as last cluster is always the cluster that contains the synapse information
sidx = np.where(idx==3)[0]
spix = reshapedStack[sidx] #this will be where you get your synapse pixels.... more scripts on how to differentiate which synapse is what coming soon ;)