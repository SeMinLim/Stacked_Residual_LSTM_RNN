
from lstm_architecture import one_hot, run_with_config

import numpy as np

import os


#--------------------------------------------
# Neural net's config.
#--------------------------------------------

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Data shaping
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series
        self.n_classes = 5  # Final output classes

        # Training
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.005
        self.training_epochs = 250
        self.batch_size = 5
        self.clip_gradients = 15.0
        self.gradient_noise_scale = None
        # Dropout is added on inputs and after each stacked layers (but not
        # between residual layers).
        self.keep_prob_for_dropout = 0.85  # **(1/3.0)

        # Linear+relu structure
        self.bias_mean = 0.3
        # I would recommend between 0.1 and 1.0 or to change and use a xavier
        # initializer
        self.weights_stddev = 0.2

        ########
        # NOTE: I think that if any of the below parameters are changed,
        # the best is to readjust every parameters in the "Training" section
        # above to properly compare the architectures only once optimised.
        ########

        # LSTM structure
        # Features count is of 9: three 3D sensors features over time
        self.n_inputs = len(X_train[0][0])
        self.n_hidden = 28  # nb of neurons inside the neural network
        # Use bidir in every LSTM cell, or not:
        self.use_bidirectionnal_cells = False

        # High-level deep architecture
        self.also_add_dropout_between_stacked_cells = False  # True
        # NOTE: values of exactly 1 (int) for those 2 high-level parameters below totally disables them and result in only 1 starting LSTM.
        # self.n_layers_in_highway = 1  # Number of residual connections to the LSTMs (highway-style), this is did for each stacked block (inside them).
        # self.n_stacked_layers = 1  # Stack multiple blocks of residual
        # layers.


#--------------------------------------------
# Dataset-specific constants and functions + loading
#--------------------------------------------

# Output classes to learn how to classify
LABELS = [
    "FOREHAND STROKE",
    "FOREHAND DRIVE",
    "FOREHAND SLICE",
    "BACKHAND DRIVE",
    "BACKHAND SLICE"
]

x_train= np.loadtxt("X_train.CSV", delimiter = ",", dtype = np.float32) #720x38 matrics
X_train_t = np.vstack([[x_train[:40,:]],[x_train[40:80,:]],[x_train[80:120,:]],[x_train[120:160,:]],
                    [x_train[160:200,:]],[x_train[200:240,:]],[x_train[240:280,:]],[x_train[280:320,:]],
                    [x_train[320:360,:]],[x_train[360:400,:]],[x_train[400:440,:]],[x_train[440:480,:]],
                    [x_train[480:520,:]],[x_train[520:560,:]],[x_train[560:600,:]],[x_train[600:640,:]],
                    [x_train[640:680,:]],[x_train[680:720,:]]])
X_train = np.transpose(X_train_t, (1,2,0))

x_test = np.loadtxt("X_test.CSV", delimiter = ",", dtype = np.float32)
X_test_t = np.vstack([[x_test[:10,:]],[x_test[10:20,:]],[x_test[20:30,:]],[x_test[30:40,:]],
                    [x_test[40:50,:]],[x_test[50:60,:]],[x_test[60:70,:]],[x_test[70:80,:]],
                    [x_test[80:90,:]],[x_test[90:100,:]],[x_test[100:110,:]],[x_test[110:120,:]],
                    [x_test[120:130,:]],[x_test[130:140,:]],[x_test[140:150,:]],[x_test[150:160,:]],
                    [x_test[160:170,:]],[x_test[170:180,:]]])
X_test = np.transpose(X_test_t,(1,2,0))

t_data_1 = np.loadtxt("Y_train.CSV", delimiter = ",", dtype = np.int32)
Y_train = t_data_1[:] - 1
y_train = one_hot(Y_train)

t_data_2 = np.loadtxt("Y_test.CSV", delimiter = ",", dtype = np.int32)
Y_test = t_data_2[:] - 1
y_test = one_hot(Y_test)
#--------------------------------------------
# Training (maybe multiple) experiment(s)
#--------------------------------------------

n_layers_in_highway = 0
n_stacked_layers = 3
trial_name = "{}x{}".format(n_layers_in_highway, n_stacked_layers)

for learning_rate in [0.001]:  # [0.01, 0.007, 0.001, 0.0007, 0.0001]:
    for lambda_loss_amount in [0.005]:
        for clip_gradients in [15.0]:
            print "learning_rate: {}".format(learning_rate)
            print "lambda_loss_amount: {}".format(lambda_loss_amount)
            print ""

            class EditedConfig(Config):
                def __init__(self, X, Y):
                    super(EditedConfig, self).__init__(X, Y)

                    # Edit only some parameters:
                    self.learning_rate = learning_rate
                    self.lambda_loss_amount = lambda_loss_amount
                    self.clip_gradients = clip_gradients
                    # Architecture params:
                    self.n_layers_in_highway = n_layers_in_highway
                    self.n_stacked_layers = n_stacked_layers

            # # Useful catch upon looping (e.g.: not enough memory)
            # try:
            #     accuracy_out, best_accuracy = run_with_config(EditedConfig)
            # except:
            #     accuracy_out, best_accuracy = -1, -1
            accuracy_out, best_accuracy, f1_score_out, best_f1_score = (
                run_with_config(EditedConfig, X_train, y_train, X_test, y_test)
            )
            print (accuracy_out, best_accuracy, f1_score_out, best_f1_score)

            with open('{}_result_HAR_6.txt'.format(trial_name), 'a') as f:
                f.write(str(learning_rate) + ' \t' + str(lambda_loss_amount) + ' \t' + str(clip_gradients) + ' \t' + str(
                    accuracy_out) + ' \t' + str(best_accuracy) + ' \t' + str(f1_score_out) + ' \t' + str(best_f1_score) + '\n\n')

            print "________________________________________________________"
        print ""
print "Done."
