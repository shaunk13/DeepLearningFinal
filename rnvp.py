from os import stat_result
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import reverse
import tensorflow_probability as tfp
from helpers import checkerboard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape

class RNVP(tf.keras.Model):
    def __init__(self, depth, prior_dim):
        super(RNVP, self).__init__()

        # prior is a distribution. In our toy example prior is a 2 dimensional gaussian with 0 mean and unit covariance
        # When we apply this to MNIST our prior is a 784 dimensional gaussian ?? (not quite sure if that's accurate
        # maybe we could make like our VAE and get our representation down to a lower dimension before sampling using rnvp)

        # mask is a checkerboard tensor. In our toy example each row of mask is an alternating [0,1...] or [1,0...] 
        # such that the dimensionality of the row matches the dimensionality of the prior and the number of rows is the depth of
        # the network ?? (not sure about why in the toy example we have depth 3/ if that depth is arbitrary)

        # t and s are themselves networks, I think we can inititalize them as standard dense networks with # of hidden layers
        # = to the length of mask 

        # based on the above, I think the only parameters we need to care about are the dimensionality of the prior and the depth
        # of the networks 

        self.prior = tfp.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(prior_dim), covariance_matrix= tf.eye(prior_dim))
        self.mask = checkerboard(prior_dim, depth*2)
        self.t = [Sequential()] * depth
        self.s = [Sequential()] * depth 
        self.hidden_size = 128

        for j in range(depth):
            self.t[j].add(tf.keras.layers.Dense(self.hidden_size,activation='relu'))
            self.s[j].add(tf.keras.layers.Dense(self.hidden_size,activation='relu'))
            for i in range(depth-2):
                self.t[j].add(tf.keras.layers.Dense(self.hidden_size,activation='relu'))
                self.s[j].add(tf.keras.layers.Dense(self.hidden_size,activation='relu'))
            self.t[j].add(tf.keras.layers.Dense(prior_dim))
            self.s[j].add(tf.keras.layers.Dense(prior_dim,activation='tanh'))

        
    def forward_transform(self,z):
        """
        Make a doc string :)
        
        Inputs: 
            z:
        Returns:
            z:
        """
        for i in range(self.depth):
            mask = self.mask[i]
            reverse_mask = 1 - mask
            z_prime = z * mask
            s_prime = self.s[i](z_prime)* reverse_mask
            t_prime = self.t[i](z_prime)* reverse_mask
            z = z_prime + (reverse_mask * (z * tf.math.exp(s_prime) + t_prime))
        return z 
    
    def reverse_transform(self,x):
        """
        Make a doc string :)

        Inputs: 
            x:
        Returns:
            x:
            sldj:
        """
        sldj = tf.zeros([x.shape[0]])
        for i in reversed(range(len(self.t))):
            mask = self.mask[i]
            reverse_mask = 1 - mask
            x_prime = x * mask
            s_prime = self.s[i](x_prime)* reverse_mask
            t_prime = self.t[i](x_prime)* reverse_mask
            x = x_prime + (reverse_mask * (x - t_prime) * tf.math.exp(-s_prime))
            sldj -= tf.math.reduce_sum(s_prime, axis=1)
        return x, sldj


    def call(self, x):
        """
        Make a doc string :)

        Inputs:
            x:
        Returns:
            ...
        """


    def loss_function(self, z, logp):
        """
        what does it do?

        Inputs:
            z:
            logp:
        Returns:
            ...
        """
        

