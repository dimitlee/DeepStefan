import tensorflow as tf
import numpy as np
import timeit

class Sampler:
    # Initialize the class
    def __init__(self, dim, coords, func, name=None):
        self.dim = dim
        self.coords = coords
        self.func = func
        self.name = name

    def sample(self, N):
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * np.random.uniform(0, 1, size=(N, self.dim))
        y = self.func(x)
        return x, y

class DataSampler:
    # Initialize the class
    def __init__(self, X, Y, name = None):
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]

    def sample(self, batch_size):
        idx = np.random.choice(self.N, batch_size, replace=True)
        X_batch = self.X[idx, :]
        Y_batch = self.Y[idx, :]
        return X_batch, Y_batch

class StefanIntegral(tf.Module):
    def __init__(self, layers_u, layers_s, ics_sampler, Ncs_sampler, integral_sampler, res_sampler):
        super(StefanIntegral, self).__init__()  # Инициализация базового класса
       # Normalization constants
        X, _ = res_sampler.sample(int(1e5))
        self.mu_X, self.sigma_X = X.mean(0), X.std(0)
        self.mu_x, self.sigma_x = self.mu_X[0], self.sigma_X[0]
        self.mu_t, self.sigma_t = self.mu_X[1], self.sigma_X[1]

        # Samplers
        self.ics_sampler = ics_sampler
        self.Ncs_sampler = Ncs_sampler
        self.integral_sampler = integral_sampler
        self.res_sampler = res_sampler

        # Initialize network weights and biases
        self.layers_u = layers_u
        self.weights_u, self.biases_u = self.initialize_NN(layers_u)
        
        self.layers_s = layers_s
        self.weights_s, self.biases_s = self.initialize_NN(layers_s)

        # Initialize loss logs
        self.loss_bcs_log = []
        self.loss_ics_log = []
        self.loss_res_log = []

        # Define optimizer with learning rate schedule
        starter_learning_rate = 1e-3
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
           initial_learning_rate=starter_learning_rate,
           decay_steps=1000,
           decay_rate=0.9,
           staircase=False
        )

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


    # Xavier initialization
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
        return tf.Variable(tf.random.normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev,
                           dtype=tf.float32)

    # Initialize network weights and biases using Xavier initialization
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

    def save_model(self, file_path):
        """Метод для сохранения весов"""
        # Сохраняем веса
        checkpoint = tf.train.Checkpoint(weights_u=self.weights_u, weights_s=self.weights_s)
        checkpoint.save(file_path)

    def load_model(self, file_path):
        """Метод для загрузки весов"""
        checkpoint = tf.train.Checkpoint(weights_u=self.weights_u, weights_s=self.weights_s)
        checkpoint.restore(file_path)


    # Evaluates the forward pass u
    def forward_pass_u(self, H):
        num_layers = len(self.layers_u)
        for l in range(0, num_layers - 2):
            W = self.weights_u[l]
            b = self.biases_u[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights_u[-1]
        b = self.biases_u[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Evaluates the forward pass s
    def forward_pass_s(self, H):
        num_layers = len(self.layers_s)
        for l in range(0, num_layers - 2):
            W = self.weights_s[l]
            b = self.biases_s[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = self.weights_s[-1]
        b = self.biases_s[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H

    # Forward pass for u
    def net_u(self, x, t):
        u = self.forward_pass_u(tf.concat([x, t], 1))
        return u

    # Forward pass for s
    def net_s(self, t):
        s = self.forward_pass_s(t)
        return s

    # Forward pass for u_x (derivative)
    def net_u_x(self, x, t):
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self.net_u(x, t)
        u_x = tape.gradient(u, x) / self.sigma_x
        return u_x

    # Forward pass for residual
    def net_r_u(self, x, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(x)
            u = self.net_u(x, t)
            u_t = tape.gradient(u, t) / self.sigma_t
            u_x = tape.gradient(u, x) / self.sigma_x
            u_xx = tape.gradient(u_x, x) / self.sigma_x
        del tape
        residual = u_t - u_xx
        return residual

    # Forward pass for Neumann condition residual: u_x(s, t) = some function
    def net_r_Nc(self, t):
        s = self.net_s(t)
        s = (s - self.mu_x) / self.sigma_x
        u_x = self.net_u_x(s, t)
        residual = u_x
        return residual
    
    # Forward pass for the integral boundary condition residual from x=0 to x=s(t)
    def net_r_integral_bc(self, t):
        # setup the start and end of the interval
        s = self.net_s(t)
        start = tf.cast((0 - self.mu_x) / self.sigma_x, tf.float32)
        end = tf.cast((s - self.mu_x) / self.sigma_x, tf.float32)
        N_STEPS = 10
        # get the datapoints along the interval
        x_quad = tf.linspace(start, end, N_STEPS)
        t_quad = tf.ones_like(x_quad) * t
        x_quad_reshaped = tf.reshape(x_quad, (-1, 1))  # x_quad_reshaped.shape = (N_STEPS * batch_size, 1)
        t_quad_reshaped = tf.reshape(t_quad, (-1, 1))  # t_quad_reshaped.shape = (N_STEPS * batch_size, 1)
        # evaluate u(x, t)
        u_quad_pred = self.net_u(x_quad_reshaped, t_quad_reshaped)
        u_quad_pred = tf.reshape(u_quad_pred, x_quad.shape)
        # get the differences between consecutive x-values (dx)
        dx = x_quad[1:] - x_quad[:-1]
        # calculate the average of consecutive u-values
        u_avg = 0.5 * (u_quad_pred[1:] + u_quad_pred[:-1])
        # multiply and sum to get the integral
        integral_approx_normalized = tf.math.reduce_sum(dx * u_avg, axis=0)
        return integral_approx_normalized


    def fetch_minibatch(self, sampler, N):
        X, Y = sampler.sample(N)
        X = (X - self.mu_X) / self.sigma_X
        return X, Y

    @tf.function
    def train_step(self, tf_dict):
        with tf.GradientTape() as tape:
            # Evaluate predictions
            self.s_pred = self.net_s(tf_dict['t_u_tf'])
            self.u_pred = self.net_u(tf_dict['x_u_tf'], tf_dict['t_u_tf'])

            # Initial condition predictions
            self.u_0_pred = self.net_u(tf_dict['x_0_tf'], tf_dict['t_0_tf'])
            self.s_0_pred = self.net_s(tf_dict['t_0_tf'])
            
            # Boundary condition predictions
            self.u_Sbc_pred = self.net_u((self.s_pred - self.mu_x) / self.sigma_x, tf_dict['t_u_tf']) # Dirichlet boundary condition u(s(t), t) = 0
            self.r_Nc_pred = self.net_r_Nc(tf_dict['t_Nc_tf']) # Neumann boundary condition u_x(s(t), t) = sqrt(3 - 2t)
            self.r_intcs_pred = self.net_r_integral_bc(tf_dict['t_intcs_tf']) # Integral type boundary condition: integral[0 to s(t)] u(x, t)dx = e(t)

            # Residual
            self.r_u_pred = self.net_r_u(tf_dict['x_r_tf'], tf_dict['t_r_tf'])

            # Initial conditions loss
            self.loss_u_0 = tf.reduce_mean(tf.square(self.u_0_pred - tf_dict['u_0_tf']))
            self.loss_s_0 = tf.reduce_mean(tf.square(self.s_0_pred - (2.0 - np.sqrt(3))))
            
            # Boundary conditions loss
            self.loss_Sbc = tf.reduce_mean(tf.square(self.u_Sbc_pred)) # Dirichlet
            self.loss_SNc = tf.reduce_mean(tf.square(self.r_Nc_pred - tf_dict['s_Nc_tf'])) # Neumann
            self.loss_intcs = tf.reduce_mean(tf.square(self.r_intcs_pred - tf_dict['u_intcs_tf']))

            # Residual loss
            self.loss_res = tf.reduce_mean(tf.square(self.r_u_pred))

            # Total loss
            self.loss_ics = self.loss_s_0 + self.loss_u_0
            self.loss_bcs = self.loss_Sbc + self.loss_SNc + self.loss_intcs
            self.loss = self.loss_bcs + self.loss_ics + self.loss_res

        # Use the individual variables directly
        trainable_variables = self.weights_u + self.biases_u + self.weights_s + self.biases_s
        gradients = tape.gradient(self.loss, trainable_variables)

        # Replace None gradients with zeros
        gradients = [grad if grad is not None else tf.zeros_like(var) for grad, var in zip(gradients, trainable_variables)]
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return self.loss, self.loss_ics, self.loss_bcs, self.loss_res

    def train(self, nIter=10000, batch_size=128):
        start_time = timeit.default_timer()
        for it in range(nIter):
            # Fetch boundary and Neumann mini-batches
            X_ics_batch, u_ics_batch = self.fetch_minibatch(self.ics_sampler, batch_size)
            X_Ncs_batch, u_Ncs_batch = self.fetch_minibatch(self.Ncs_sampler, batch_size)
            X_intcs_batch, u_intcs_batch = self.fetch_minibatch(self.integral_sampler, batch_size)

            # Fetch residual mini-batch
            X_res_batch, _ = self.fetch_minibatch(self.res_sampler, batch_size)

            # Define a dictionary for associating data with tensors
            tf_dict = {
                'x_u_tf':       tf.convert_to_tensor(X_res_batch[:, 0:1], dtype=tf.float32),
                't_u_tf':       tf.convert_to_tensor(X_res_batch[:, 1:2], dtype=tf.float32),
                'x_0_tf':       tf.convert_to_tensor(X_ics_batch[:, 0:1], dtype=tf.float32),
                't_0_tf':       tf.convert_to_tensor(X_ics_batch[:, 1:2], dtype=tf.float32),
                'u_0_tf':       tf.convert_to_tensor(u_ics_batch,         dtype=tf.float32),
                't_Nc_tf':      tf.convert_to_tensor(X_Ncs_batch[:, 1:2], dtype=tf.float32),
                's_Nc_tf':      tf.convert_to_tensor(u_Ncs_batch,         dtype=tf.float32),
                't_intcs_tf':   tf.convert_to_tensor(X_intcs_batch[:, 1:2], dtype=tf.float32),
                'u_intcs_tf':   tf.convert_to_tensor(u_intcs_batch,       dtype=tf.float32),
                'x_r_tf':       tf.convert_to_tensor(X_res_batch[:, 0:1], dtype=tf.float32),
                't_r_tf':       tf.convert_to_tensor(X_res_batch[:, 1:2], dtype=tf.float32),
            }

            # Perform a training step
            loss_value, loss_ics_value, loss_bcs_value, loss_res_value = self.train_step(tf_dict)

            # Log and print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                self.loss_ics_log.append(loss_ics_value.numpy())
                self.loss_bcs_log.append(loss_bcs_value.numpy())
                self.loss_res_log.append(loss_res_value.numpy())

                print('It: %d, Loss: %.3e, Loss_ics: %.3e, Loss_bcs: %.3e, Loss_res: %.3e, Time: %.2f' %
                      (it, loss_value.numpy(), loss_ics_value.numpy(), loss_bcs_value.numpy(), loss_res_value.numpy(), elapsed))
                start_time = timeit.default_timer()

    # Predictions for u
    def predict_u(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        x = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        u_star = self.net_u(x, t)
        return u_star.numpy()

    # Predictions for s
    def predict_s(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        t = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        s_star = self.net_s(t)
        return s_star.numpy()

    # Predictions for u_x
    def predict_u_x(self, X_star):
        X_star = (X_star - self.mu_X) / self.sigma_X
        x = tf.convert_to_tensor(X_star[:, 0:1], dtype=tf.float32)
        t = tf.convert_to_tensor(X_star[:, 1:2], dtype=tf.float32)
        u_x_pred = self.net_u_x(x, t)
        return u_x_pred.numpy()

    # Save model weights
    def save_weights(self, filepath):
        # Создаём объект Checkpoint для сохранения весов
        checkpoint = tf.train.Checkpoint(weights_u=self.weights_u,
                                         biases_u=self.biases_u,
                                         weights_s=self.weights_s,
                                         biases_s=self.biases_s)
        checkpoint.save(filepath)

    # Load model weights
    def load_weights(self, filepath):
        checkpoint = tf.train.Checkpoint(weights_u=self.weights_u,
                                         biases_u=self.biases_u,
                                         weights_s=self.weights_s,
                                         biases_s=self.biases_s)
        checkpoint.restore(filepath).expect_partial()