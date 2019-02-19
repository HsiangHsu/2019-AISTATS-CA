"""
FG-net core utilities

Authors: hsianghsu, flavio@seas.harvard.edu
"""

import tensorflow as tf
import numpy as np
import scipy as sp

# loss function
def create_loss_svd(f_out,g_out,clip_min = np.float32(-10000),clip_max=np.float32(10000)):
    """
    Create the loss function that will be minimized by the fg-net. Many options exist.
    The implementation below uses the 1-Schatten norm from the derivation. It might slow.
    
    
    Inputs: f_out and g_out, which are tensors of the same shape produced by the outut
            of the f and g nets. Assumes that they are of the form (#batches, #output).
            
    Outputs: returns objective
    """
 
    # number of samples in batch
    nBatch = tf.cast(tf.shape(f_out)[0],tf.float32)
    
    # we clip f to avoid runaway arguments
    f_clip = tf.clip_by_value(f_out,clip_min,clip_max)
    #f_clip = f_out
    
    # create correlation matrices
    corrF = tf.matmul(tf.transpose(f_clip),f_clip)/nBatch
    corrFG = tf.matmul(tf.transpose(f_clip),g_out)/nBatch
    
    # Second moment of g
    sqG = tf.reduce_sum(tf.reduce_mean(tf.square(g_out),0))
    
    # compute svd in objective
    n = tf.shape(corrF)[0] 
    
    #correction term
    epsilon =1e-4
    
    invCorrF = tf.matrix_inverse(corrF+epsilon*tf.eye(n), adjoint=True) #check
    
    
    prodGiFG = tf.matmul(tf.matmul(tf.transpose(corrFG),invCorrF),corrFG)
    
    s,v = tf.self_adjoint_eig(prodGiFG)
    #s,u,v = tf.svd(prodGiFG)
    
    schatNorm = tf.reduce_sum(tf.sqrt(tf.abs(s)))
    
    
    # define objective
    objective = sqG - 2*schatNorm #+ tf.trace(corrF)#.3*tf.reduce_sum(tf.square((corrF-tf.eye(n))))
    
    #return objective
    return objective,schatNorm


# create simple Feed Foreword NN for experiments
def SimpleNet(inputShape,structure = [40, 40, 20, 15],name="simpleNet"):
    """
    Creates a simple feed forward neural network.
    
    Input:  the standard input will have 1 dimension, You could replace this to call any 
            other constructor of a neural network. Structure defines the shape. The final
            layer will be d
            
    Output: Three variables that can be used to control the network:
            - x_input: input to the NN
            - f_out: output at the final layer
            - keepProb: dropout probability
    """
    
    # create name scope
    with tf.variable_scope(name):

        # creat input to network
        x_input = tf.placeholder(tf.float32,shape=[None,inputShape])

        # initialize with no dropout
        keepProb = tf.placeholder_with_default(1.0,[])

        # create list of intermediate outputs
        midOutputs = []

        # current layer being built
        layer = 0

        # total number of layers
        numLayers = len(structure)

        # create input layer
        with tf.variable_scope("input_layer"):
            numGates = structure[0]
            weights = tf.get_variable("weights",[inputShape,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
            bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

            output = tf.nn.tanh(tf.matmul(x_input,weights)+bias)
            midOutputs.append(tf.nn.dropout(output,keepProb))
        layer+=1

        # create intermediate layers
        for i in range(1,numLayers-1):
            with tf.variable_scope("middle_layer_"+str(i)):
                numGates = structure[i]
                numInputs = structure[i-1]

                weights = tf.get_variable("weights",[numInputs,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
                bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

                output = tf.nn.tanh(tf.matmul(midOutputs[layer-1],weights)+bias)
                midOutputs.append(tf.nn.dropout(output,keepProb))

            layer+=1

        # create output layer
        with tf.variable_scope("output_layer"):
            numGates = structure[-1]
            numInputs = structure[-2]
            weights = tf.get_variable("weights",[numInputs,numGates],
                                      initializer=tf.truncated_normal_initializer(stddev=.1))
            bias = tf.get_variable("biases",[numGates],initializer=tf.constant_initializer(.1))

            # no relu in output
            #output = tf.nn.relu(tf.matmul(midOutputs[layer-1],weights)+bias)
            output = tf.nn.tanh(tf.matmul(midOutputs[layer-1],weights)+bias)

            # create constant 1 and append -- this will represent the constant function
            const1 = tf.fill([tf.shape(output)[0],1],np.float32(1))

            final_output = tf.concat([const1,output],axis=1)
            
            #f_clip = tf.clip_by_value(final_output,-10,10)
    # return values
    return (x_input, final_output,keepProb)
        
    
# def computeMetrics(F,G):
#     corrF = F.transpose().dot(F)/F.shape[0]
#    corrFG = F.transpose().dot(G)/G.shape[0]
#    corrG = G.transpose().dot(G)/G.shape[0]
#    uF,sF,vF = np.linalg.svd(corrF)
#    invFsqrt = (uF*(sF**(-.5))).dot(vF)
#    u,s,v = np.linalg.svd(invFsqrt.dot(corrFG))
#    Anorm = v.transpose().dot(u.transpose()).dot(invFsqrt)
#    Bnorm = np.diag(np.diag(corrG)**(-.5))
    # check if Anorm whitens
#    wF = F.dot(Anorm.transpose())
#    wG = G.dot(Bnorm)
#    newCorr = wF.transpose().dot(wF)/F.shape[0]
#    trueCorr = wF.transpose().dot(wG)/G.shape[0]
    

    
#    return (trueCorr,corrG,newCorr,Anorm,Bnorm,wF,wG)
    
    
# Returns whitening matrix and mean removal for F, G
# assumes first entry of F and G is function of all ones
# input:    F = f ouput for *training* dataset
#           G = g ouput for *training* dataset
# output:   A,a_mean,B,b_mean
#           (a_mean,A) such that Fnorm. = (F-a_mean).dot(A) is white
#           (b_mean,B) such that Gnorm.= (G-b_mean).dot(B) is white
#           np.diag(Fnorm.transpose().dot(Gnorm)/Gnorm.shape[0]) will be the PICs

def normalizeFG(F,G):
    # Values for G
    Gs = G[:,1:]
    b_mean = Gs.mean(axis=0)
    Gs = Gs - b_mean
    corrG = Gs.transpose().dot(Gs)/Gs.shape[0]
    U,v,_ = np.linalg.svd(corrG)
    corrG_sqrt_inv = (U*(v)**(-.5)).dot(U.transpose())
    
    b_mean = np.concatenate(([0],b_mean))
    B = sp.linalg.block_diag(1,corrG_sqrt_inv)
    
    nG = (G-b_mean).dot(B)

    # values for F
    Fs = F[:,1:]
    a_mean = Fs.mean(axis=0)
    Fs = Fs - a_mean
    corrF = Fs.transpose().dot(Fs)/Fs.shape[0]
    U,v,_ = np.linalg.svd(corrF)
    corrF_sqrt_inv = (U*(v)**(-.5)).dot(U.transpose())
    

    a_mean = np.concatenate(([0],a_mean))
    A = sp.linalg.block_diag(1,corrF_sqrt_inv)
    
    nF = (F-a_mean).dot(A)
    
    # Create proper normalization
    U,s,V = np.linalg.svd(nF.transpose().dot(nG)/G.shape[0])

    return A.dot(U),a_mean,B.dot(V.transpose()),b_mean
