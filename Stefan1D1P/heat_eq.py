import tensorflow as tf
import numpy as np
import timeit


class Heat(tf.Module):
    # Constructor
    def __init__(self, layers, ics_sampler, bcls_sampler, bcrs_sampler, res_sampler, alpha):
        super(Heat, self).__init__()
        # Normalization constants
        X, _ = res_sampler.sample(int(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]
        # Samplers
        self.ics_sampler = ics_sampler
        self.bcls_sampler = bcls_sampler
        self.bcrs_sampler = bcrs_sampler
        self.res_sampler = res_sampler
        # Initialize network weights and biases
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        self.alpha = tf.constant(alpha, dtype=tf.float32)
        # Initialize loss logs
        self.loss_bcls_log = []
        self.loss_bcrs_log = []
        self.loss_ics_log = []
        self.loss_res_log = []
        # Define optimizer and decaying learning rate
        starter_learning_rate = 1e-4
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=starter_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=False
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    
    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)
    
    # Initialize network
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    # Evaluates the forward pass
    def forward_pass(self, H):
        num_layers = len(self.layers)
        for l in range(0, num_layers - 2):
            W = self.weights[l]
            b = self.biases[l]
            try:
                mult = tf.matmul(H, W)
                H = tf.tanh(tf.add(mult, b))
            except ValueError as ex:
                print(f"{mult.shape = }")
                print(f"{b.shape = }")
                raise ex
        W = self.weights[-1]
        b = self.biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    # Forward pass
    def net(self, x, t):
        u = self.forward_pass(tf.concat([x, t], 1))
        return u
    
    # Forward pass for residual
    def net_r(self, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(x)
            u = self.net(x, t)
            u_t = tape.gradient(u, t) / self.sigma_t
            u_x = tape.gradient(u, x) / self.sigma_x
            u_xx = tape.gradient(u_x, x) / self.sigma_x
        del tape
        residual = u_t - (self.alpha**2)*u_xx
        return residual
    
    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y
    
    @tf.function
    def train_step(self, tf_dict):
        with tf.GradientTape() as tape:
            # Evaluate predictions
            self.u_pred = self.net(tf_dict['x_u_tf'], tf_dict['t_u_tf'])

            self.u_ic_pred = self.net(tf_dict['x_ic_tf'], tf_dict['t_ic_tf'])
            self.u_bcl_pred = self.net(tf_dict['x_bcl_tf'], tf_dict['t_bcl_tf'])
            self.u_bcr_pred = self.net(tf_dict['x_bcr_tf'], tf_dict['t_bcr_tf'])

            self.r_u_pred = self.net_r(tf_dict['x_r_tf'], tf_dict['t_r_tf'])

            # Boundary and initial loss
            self.loss_u_ic = tf.reduce_mean(tf.square(self.u_ic_pred - tf_dict['u_ic_tf']))
            self.loss_u_bcl = tf.reduce_mean(tf.square(self.u_bcl_pred - tf_dict['u_bcl_tf']))
            self.loss_u_bcr = tf.reduce_mean(tf.square(self.u_bcr_pred - tf_dict['u_bcr_tf']))

            # Residual loss
            self.loss_u_res = tf.reduce_mean(tf.square(self.r_u_pred))

            # Total loss
            self.loss_u_bc = self.loss_u_bcl + self.loss_u_bcr
            self.loss = self.loss_u_ic + self.loss_u_bc + self.loss_u_res

        # Use the individual variables directly
        trainable_variables = self.weights + self.biases
        gradients = tape.gradient(self.loss, trainable_variables)

        # Replace None gradients with zeros
        gradients = [grad if grad is not None else tf.zeros_like(var) for grad, var in zip(gradients, trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return self.loss, self.loss_u_ic, self.loss_u_bcl, self.loss_u_bcr, self.loss_u_res
            
    def train(self, n_iter=10000, batch_size=128):
        start_time = timeit.default_timer()
        for it in range(n_iter):
            # Fetch boundary and Neumann mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size)
            X_bcls_batch, u_bcls_batch = self.fetch_minibatch(self.bcls_sampler, batch_size)
            X_bcrs_batch, u_bcrs_batch = self.fetch_minibatch(self.bcrs_sampler, batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating data with tensors
            tf_dict = {
                'x_u_tf': tf.convert_to_tensor(X_res_batch[:, 0:1], dtype=tf.float32),
                't_u_tf': tf.convert_to_tensor(X_res_batch[:, 1:2], dtype=tf.float32),
                'x_ic_tf': tf.convert_to_tensor(X_ics_batch[:, 0:1], dtype=tf.float32),
                't_ic_tf': tf.convert_to_tensor(X_ics_batch[:, 1:2], dtype=tf.float32),
                'u_ic_tf': tf.convert_to_tensor(u_ics_batch, dtype=tf.float32),
                'x_bcl_tf': tf.convert_to_tensor(X_bcls_batch[:, 0:1], dtype=tf.float32),
                't_bcl_tf': tf.convert_to_tensor(X_bcls_batch[:, 1:2], dtype=tf.float32),
                'u_bcl_tf': tf.convert_to_tensor(u_bcls_batch, dtype=tf.float32),
                'x_bcr_tf': tf.convert_to_tensor(X_bcrs_batch[:, 0:1], dtype=tf.float32),
                't_bcr_tf': tf.convert_to_tensor(X_bcrs_batch[:, 1:2], dtype=tf.float32),
                'u_bcr_tf': tf.convert_to_tensor(u_bcrs_batch, dtype=tf.float32),
                'x_r_tf': tf.convert_to_tensor(X_res_batch[:, 0:1], dtype=tf.float32),
                't_r_tf': tf.convert_to_tensor(X_res_batch[:, 1:2], dtype=tf.float32),
            }

            # Perform a training step
            loss_value, loss_ics_value, loss_bcls_value, loss_bcrs_value, loss_res_value = self.train_step(tf_dict)

            # Log and print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                self.loss_ics_log.append(loss_ics_value.numpy())
                self.loss_bcls_log.append(loss_bcls_value.numpy())
                self.loss_bcrs_log.append(loss_bcrs_value.numpy())
                self.loss_res_log.append(loss_res_value.numpy())

                print('It: %d, Loss: %.3e, Loss_ics: %.3e, Loss_bcls: %.3e, Loss_bcrs: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value.numpy(), loss_ics_value.numpy(), loss_bcls_value.numpy(), loss_bcrs_value.numpy(), loss_res_value.numpy(), elapsed))
                start_time = timeit.default_timer()
                
    def predict(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        x = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        u_star = self.net(x, t)
        return u_star.numpy()
    
    # Save model weights
    def save_weights(self, filepath):
        # Создаём объект Checkpoint для сохранения весов
        checkpoint = tf.train.Checkpoint(weights=self.weights,
                                         biases=self.biases)
        checkpoint.save(filepath)

    # Load model weights
    def load_weights(self, filepath):
        checkpoint = tf.train.Checkpoint(weights=self.weights,
                                         biases=self.biases)
        checkpoint.restore(filepath).expect_partial()