from keras.engine import Layer, InputSpec
from keras import initializations
from keras import backend as K


class BatchNormalization(Layer):
    '''Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Arguments
        epsilon: small float > 0. Fuzz parameter.
        axis: integer, axis along which to normalize. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://jmlr.org/proceedings/papers/v37/ioffe15.html)

    # NOTICE
      This code is based on [keras/normalization.py at aea00258e7c8548bc0b9b91731fe606ce79509f0](https://github.com/fchollet/keras/blob/aea00258e7c8548bc0b9b91731fe606ce79509f0/keras/layers/normalization.py).
      See [keras/LICENSE](https://github.com/fchollet/keras/blob/aea00258e7c8548bc0b9b91731fe606ce79509f0/LICENSE).
    '''
    def __init__(self, epsilon=1e-6, axis=-1, momentum=0.9,
                 weights=None, beta_init='zero', gamma_init='one', **kwargs):
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.epsilon = epsilon
        self.axis = axis
        self.momentum = momentum
        self.initial_weights = weights
        self.uses_learning_phase = True
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        self.running_mean = K.zeros(shape,
                                    name='{}_running_mean'.format(self.name))
        self.running_std = K.ones(shape,
                                  name='{}_running_std'.format(self.name))
        self.non_trainable_weights = [self.running_mean, self.running_std]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # case: train mode (uses stats of the current batch)
        mean = K.mean(x, axis=reduction_axes)
        brodcast_mean = K.reshape(mean, broadcast_shape)
        std = K.mean(K.square(x - brodcast_mean) + self.epsilon, axis=reduction_axes)
        std = K.sqrt(std)
        brodcast_std = K.reshape(std, broadcast_shape)
        mean_update = self.momentum * self.running_mean + (1 - self.momentum) * mean
        std_update = self.momentum * self.running_std + (1 - self.momentum) * std
        self.updates = [(self.running_mean, mean_update),
                        (self.running_std, std_update)]
        x_normed = (x - brodcast_mean) / (brodcast_std + self.epsilon)

        # case: test mode (uses running averages)
        brodcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
        brodcast_running_std = K.reshape(self.running_std, broadcast_shape)
        x_normed_running = ((x - brodcast_running_mean) / (brodcast_running_std + self.epsilon))

        # pick the normalized form of x corresponding to the training phase
        x_normed = K.in_train_phase(x_normed, x_normed_running)
        out = K.reshape(self.gamma, broadcast_shape) * x_normed + K.reshape(self.beta, broadcast_shape)

        return out

    def get_config(self):
        config = {
            "epsilon": self.epsilon,
            "axis": self.axis,
            "momentum": self.momentum
        }
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
