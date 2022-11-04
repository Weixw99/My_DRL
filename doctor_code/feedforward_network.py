
import numpy as np
import tensorflow as tf

def feedforward_network(inputState, inputSize, outputSize, num_fc_layers, depth_fc_layers, tf_datatype):  

    
    #vars
    intermediate_size=depth_fc_layers
    reuse= False
    initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf_datatype)
    fc = tf.contrib.layers.fully_connected  

#     print("#####inputshape####",inputState.shape)
    # make hidden layers
    for i in range(num_fc_layers):
        if(i==0):
            fc_i = fc(inputState, num_outputs=intermediate_size, activation_fn=None, 
                    weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
        else:
            fc_i = fc(h_i, num_outputs=intermediate_size, activation_fn=None, 
                    weights_initializer=initializer, biases_initializer=initializer, reuse=reuse, trainable=True)
        h_i = tf.nn.relu(fc_i)
  
    # make output layer
    z=fc(h_i, num_outputs=outputSize, activation_fn=None, weights_initializer=initializer, 
        biases_initializer=initializer, reuse=reuse, trainable=True)
    return z





####    inputs: A tensor of at least rank 2 and static value for the last dimension; i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
####    num_outputs: Integer or long, the number of output units in the layer.
####    activation_fn: Activation function. The default value is a ReLU function.Explicitly set it to None to skip it and maintain a linear activation.
####    normalizer_fn: Normalization function to use instead of `biases`. If `normalizer_fn` is provided then `biases_initializer` and
####    `biases_regularizer` are ignored and `biases` are not created nor added.default set to None for no normalizer function
####    normalizer_params: Normalization function parameters.
####    weights_initializer: An initializer for the weights.
####    weights_regularizer: Optional regularizer for the weights.
####    biases_initializer: An initializer for the biases. If None skip biases.
####    biases_regularizer: Optional regularizer for the biases.
####    reuse: Whether or not the layer and its variables should be reused. To be able to reuse the layer scope must be given.
####    variables_collections: Optional list of collections for all the variables or a dictionary containing a different list of collections per variable.
####    outputs_collections: Collection to add the outputs.
####    trainable: If `True` also add variables to the graph collection `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
####    scope: Optional scope for variable_scope.

