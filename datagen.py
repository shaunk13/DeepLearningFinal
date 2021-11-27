import numpy as np
import tensorflow as tf

def datagenerator(dims, num_examples, num_samples, mix_number):
    """
    Takes in dimensions of sample points we wish to generate,
    num_examples as number of toy distributions, num_samples as
    number of datapoints per toy distribution, and mix_number as
    number of simple gaussians we combine in our mix
    """

    mus = np.array(tf.random.uniform([num_examples,mix_number,dims], minval= 0, maxval= 100))
    covs = np.array(tf.random.uniform([num_examples,mix_number,dims,dims], minval= 0, maxval= 10))
    probs = np.array(tf.random.uniform([num_examples,mix_number],minval=0, maxval=1))
    probs = probs/np.expand_dims((np.sum(probs,axis=1)),axis=1)
    points = np.zeros([num_examples,num_samples,dims])


    for i in range(num_examples):
        # select a number from range num_mus
        # take that mu and the related covs and sample a data point 
        # repeat for num_points 
        for j in range(num_samples):
            index = np.random.choice(range(mix_number), p=probs[i,:])
            mu = mus[i,index,:]
            cov = covs[i,index,:,:]
            points[i,j,:] = np.random.multivariate_normal(mean= mu, cov=cov)
    return(points, mus, covs, probs)


