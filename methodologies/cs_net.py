import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as torch_optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timeit import default_timer as timer
import warnings
# from sklearn.model_selection._split import StratifiedShuffleSplit
# from experiments.experimental_design import experimental_design

torch.manual_seed(42)


class CostInsensitiveDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CostSensitiveDataset(Dataset):
    def __init__(self, X, y, w):
        self.X = X
        self.y = y
        self.w = w

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.w[idx]


class CSNeuralNetwork(nn.Module):
    def __init__(self, n_inputs, cost_sensitive=False, obj='ce', lambda1=0, lambda2=0, n_neurons=16):
        # TODO:
        #   One hidden layer - tune number of neurons as hyperparameter!
        #   Compare relu with hyperbolic tangent
        #   Only add BatchNorm in regularized version
        #   Add more arguments to object initialization (e.g. objective function)
        super().__init__()

        self.n_inputs = n_inputs
        self.cost_sensitive = (obj == 'weightedce' or obj == 'aec')
        self.obj = obj

        self.lin_layer1 = nn.Linear(n_inputs, n_neurons)
        self.final_layer = nn.Linear(n_neurons, 1)
        self.sigmoid = nn.Sigmoid()

        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, x):
        x = self.lin_layer1(x)
        # x = F.relu(x)
        # Todo: check difference with tanh?
        x = torch.tanh(x)
        x = self.final_layer(x)
        x = self.sigmoid(x)

        return x

    def model_train(self, model, x_train, y_train, x_val, y_val, cost_matrix_train=None, cost_matrix_val=None,
                    n_epochs=500, batch_size=2 ** 10, verbose=True):

        early_stopping_criterion = 25

        if self.cost_sensitive:
            train_ds = CostSensitiveDataset(torch.from_numpy(x_train).float(),
                                            torch.from_numpy(y_train[:, None]).float(),
                                            torch.from_numpy(cost_matrix_train))
            val_ds = CostSensitiveDataset(torch.from_numpy(x_val).float(),
                                          torch.from_numpy(y_val[:, None]).float(),
                                          torch.from_numpy(cost_matrix_val))
        else:
            train_ds = CostInsensitiveDataset(torch.from_numpy(x_train).float(),
                                              torch.from_numpy(y_train[:, None]).float())
            val_ds = CostInsensitiveDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val[:, None]).float())
            criterion = nn.BCELoss()

        optimizer = torch_optim.Adam(model.parameters(), lr=0.001)  # Todo: larger learning rate?

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=int(batch_size / 4), shuffle=True)

        best_val_loss = float("Inf")

        epochs_not_improved = 0

        for epoch in range(n_epochs):
            start = timer()

            running_loss = 0.0

            # Training
            model.train()
            for i, data in enumerate(train_dl):
                if self.cost_sensitive:
                    inputs, labels, cost_matrix_batch = data
                else:
                    inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                if self.obj == 'ce':
                    loss = criterion(outputs, labels)
                elif self.obj == 'weightedce':
                    misclass_cost_batch = torch.zeros((len(labels), 1), dtype=torch.double)
                    misclass_cost_batch[labels == 0] = cost_matrix_batch[:, 1, 0][:, None][labels == 0]
                    misclass_cost_batch[labels == 1] = cost_matrix_batch[:, 0, 1][:, None][labels == 1]

                    loss = nn.BCELoss(weight=misclass_cost_batch)(outputs, labels)
                elif self.obj == 'aec':
                    loss = self.expected_cost(outputs, labels, cost_matrix_batch)
                else:
                    raise Exception('Objective function not recognized')

                # Add regularization
                model_params = torch.cat([params.view(-1) for params in model.parameters()])
                l1_regularization = self.lambda1 * torch.norm(model_params, 1)
                # print('l1 regularization = %.4f' % l1_regularization)
                l2_regularization = self.lambda2 * torch.norm(model_params, 2)**2  # torch.norm returns the square root
                # print('l2 regularization = %.4f' % l2_regularization)
                loss += l1_regularization + l2_regularization

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation check
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for val_i, val_data in enumerate(val_dl):
                    if self.obj == 'ce':
                        val_inputs, val_labels = val_data
                        val_outputs = model(val_inputs)
                        val_loss = criterion(val_outputs, val_labels)

                    elif self.obj == 'weightedce':
                        val_inputs, val_labels, val_cost_matrix = val_data
                        val_outputs = model(val_inputs)

                        misclass_cost_val = torch.zeros((len(val_labels), 1), dtype=torch.double)
                        misclass_cost_val[val_labels == 0] = val_cost_matrix[:, 1, 0][:, None][val_labels == 0]
                        misclass_cost_val[val_labels == 1] = val_cost_matrix[:, 0, 1][:, None][val_labels == 1]

                        val_loss = nn.BCELoss(weight=misclass_cost_val)(val_outputs, val_labels)

                    elif self.obj == 'aec':
                        val_inputs, val_labels, val_cost_matrix = val_data
                        val_outputs = model(val_inputs)
                        val_loss = self.expected_cost(val_outputs, val_labels, val_cost_matrix)

                    total_val_loss += val_loss

            end = timer()

            if total_val_loss < best_val_loss:

                # Is improvement large enough?
                # If difference in val_loss is < 10**-1  # Todo: increase?
                if best_val_loss - total_val_loss < 10**-1:
                    epochs_not_improved += 1
                    # Todo: delete next line
                    # print('\t\tDifference: {}'.format(best_val_loss - total_val_loss))
                    if epochs_not_improved > early_stopping_criterion:
                        print(
                            '\t\tEarly stopping criterion reached: validation loss not significantly improved for {}'
                            ' epochs.'.format(
                                epochs_not_improved - 1))
                        print('\t\tInsufficient improvement in validation loss')
                        break
                else:
                    epochs_not_improved = 0

                best_val_loss = total_val_loss

                checkpoint = {
                    'epoch': epoch + 1,
                    'best validation loss': best_val_loss,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}
                torch.save(checkpoint, 'checkpoint')

                if verbose:
                    if epoch % 1 == 0:
                        print('\t\t[Epoch %d]\tloss: %.8f\tval_loss: %.8f\tTime [s]: %.2f\tModel saved!' % (
                        epoch + 1, running_loss / len(train_ds), total_val_loss / len(val_ds) / 4, end-start))

            else:
                epochs_not_improved += 1
                if epochs_not_improved > early_stopping_criterion:
                    print('\t\tEarly stopping criterion reached: validation loss not significantly improved for {}'
                          ' epochs.'.format(
                        epochs_not_improved - 1))
                    break

                if verbose:
                    if epoch % 10 == 9:
                        print('\t\t[Epoch %d]\tloss: %.8f\tval_loss: %.8f\tTime [s]: %.2f' % (
                            epoch + 1, running_loss / len(train_ds), total_val_loss / len(val_ds) / 4, end-start))

        best_checkpoint = torch.load('checkpoint')
        model.load_state_dict(best_checkpoint['model'])

        if verbose:
            print('\tFinished training! Best validation loss at epoch %d (loss: %.8f)\n'
                  % (best_checkpoint['epoch'], best_val_loss / len(val_ds) / 4))

        if best_checkpoint['epoch'] > (n_epochs - early_stopping_criterion):
            warnings.warn('Number of epochs might have to be increased!')

        return model

    def model_predict(self, model, X_test):
        y_pred = torch.zeros(len(X_test)).float()

        test_ds = CostInsensitiveDataset(torch.from_numpy(X_test).float(), y_pred)  # Amounts only needed for loss

        test_dl = DataLoader(test_ds, batch_size=X_test.shape[0])

        preds = []

        model.eval()

        with torch.no_grad():
            for x, _ in test_dl:
                prob = model(x)
                preds.append(prob.flatten())

        return preds[0].numpy()  # TODO: not too clean ...

    def expected_cost(self, output, target, cost_matrix):

        ec = target * (output * cost_matrix[:, 1, 1] + (1 - output) * cost_matrix[:, 0, 1]) \
              + (1 - target) * (output * cost_matrix[:, 1, 0] + (1 - output) * cost_matrix[:, 0, 0])

        return ec.mean()

    def tune(self, l1, lambda1_list, l2, lambda2_list, neurons_list, x_train, y_train, cost_matrix_train, x_val, y_val,
             cost_matrix_val):

        results = np.ones((3, len(neurons_list)))
        results[0, :] = neurons_list

        for i, n_neurons in enumerate(neurons_list):
            print('Number of neurons: {}'.format(n_neurons))

            if l1:
                self.lambda2 = 0
                losses_list_l1 = []
                for lambda1 in lambda1_list:
                    net = CSNeuralNetwork(n_inputs=x_train.shape[1], cost_sensitive=self.cost_sensitive, obj=self.obj,
                                          lambda1=lambda1, n_neurons=n_neurons)

                    net = net.model_train(net, x_train, y_train, x_val, y_val,
                                          cost_matrix_train=cost_matrix_train, cost_matrix_val=cost_matrix_val)

                    scores_val = net.model_predict(net, x_val)

                    # Evaluate loss (without regularization term!)
                    net.lambda1 = 0
                    if self.obj == 'ce':
                        eps = 1e-9  # small value to avoid log(0)
                        ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))
                        val_loss = ce.mean()
                    elif self.obj == 'weightedce':
                        eps = 1e-9  # small value to avoid log(0)
                        ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))

                        cost_misclass = np.zeros(len(y_val))
                        cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                        cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                        weighted_ce = cost_misclass * ce
                        val_loss = weighted_ce.mean()
                    elif self.obj == 'aec':
                        def aec_val(scores, y_true):
                            ec = y_true * (
                                 scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1])\
                                 + (1 - y_true) * (
                                 scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                            return ec.mean()

                        aec = aec_val(scores_val, y_val)
                        val_loss = aec
                    print('\t\tLambda l1 = %.4f;\tLoss = %.5f' % (lambda1, val_loss))
                    losses_list_l1.append(val_loss)
                lambda1_opt = lambda1_list[np.argmin(losses_list_l1)]
                print('\tOptimal lambda = %.4f' % lambda1_opt)
                self.lambda1 = lambda1_opt

                results[1, i] = lambda1_opt
                results[2, i] = np.min(losses_list_l1)

            elif l2:
                self.lambda1 = 0
                losses_list_l2 = []
                for lambda2 in lambda2_list:
                    net = CSNeuralNetwork(n_inputs=x_train.shape[1], cost_sensitive=self.cost_sensitive, obj=self.obj,
                                          lambda2=lambda2, n_neurons=n_neurons)

                    net = net.model_train(net, x_train, y_train, x_val, y_val,
                                          cost_matrix_train=cost_matrix_train, cost_matrix_val=cost_matrix_val)

                    scores_val = net.model_predict(net, x_val)

                    # Evaluate loss (without regularization term!)
                    net.lambda2 = 0
                    if self.obj == 'ce':
                        eps = 1e-9
                        ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))
                        val_loss = ce.mean()
                    elif self.obj == 'weightedce':
                        eps = 1e-9
                        ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))

                        cost_misclass = np.zeros(len(y_val))
                        cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                        cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                        weighted_ce = cost_misclass * ce
                        val_loss = weighted_ce.mean()
                    elif self.obj == 'aec':
                        def aec_val(scores, y_true):
                            ec = y_true * (
                                 scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1])\
                                 + (1 - y_true) * (
                                 scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                            return ec.mean()

                        aec = aec_val(scores_val, y_val)
                        val_loss = aec
                    print('\t\tLambda l2 = %.4f;\tLoss = %.5f' % (lambda2, val_loss))
                    losses_list_l2.append(val_loss)
                lambda2_opt = lambda2_list[np.argmin(losses_list_l2)]
                print('\tOptimal lambda = %.4f' % lambda2_opt)
                self.lambda2 = lambda2_opt

                results[1, i] = lambda2_opt
                results[2, i] = np.min(losses_list_l2)

            else:
                self.lambda1 = 0
                self.lambda2 = 0
                net = CSNeuralNetwork(n_inputs=x_train.shape[1], cost_sensitive=self.cost_sensitive, obj=self.obj,
                                      n_neurons=n_neurons)

                net = net.model_train(net, x_train, y_train, x_val, y_val, cost_matrix_train=cost_matrix_train,
                                      cost_matrix_val=cost_matrix_val, verbose=True)

                scores_val = net.model_predict(net, x_val)

                if self.obj == 'ce':
                    eps = 1e-9
                    ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))
                    val_loss = ce.mean()
                elif self.obj == 'weightedce':
                    eps = 1e-9
                    ce = - (y_val * np.log(scores_val + eps) + (1 - y_val) * np.log(1 - scores_val + eps))

                    cost_misclass = np.zeros(len(y_val))
                    cost_misclass[y_val == 0] = cost_matrix_val[:, 1, 0][y_val == 0]
                    cost_misclass[y_val == 1] = cost_matrix_val[:, 0, 1][y_val == 1]

                    weighted_ce = cost_misclass * ce
                    val_loss = weighted_ce.mean()
                elif self.obj == 'aec':
                    def aec_val(scores, y_true):
                        ec = y_true * (
                             scores * cost_matrix_val[:, 1, 1] + (1 - scores) * cost_matrix_val[:, 0, 1]) \
                             + (1 - y_true) * (
                             scores * cost_matrix_val[:, 1, 0] + (1 - scores) * cost_matrix_val[:, 0, 0])

                        return ec.mean()

                    aec = aec_val(scores_val, y_val)
                    val_loss = aec
                print('\t\tNumber of neurons = %i;\tLoss = %.5f' % (n_neurons, val_loss))
                results[2, i] = val_loss

        # Assign best settings
        opt_ind = np.argmin(results[2, :])
        opt_n_neurons = int(results[0, opt_ind])
        print('Optimal number of neurons: {}'.format(opt_n_neurons))
        if l1:
            self.lambda1 = results[1, opt_ind]
            print('Optimal l1: {}'.format(self.lambda1))
        if l2:
            self.lambda2 = results[1, opt_ind]
            print('Optimal l2: {}'.format(self.lambda2))

        return CSNeuralNetwork(self.n_inputs, self.cost_sensitive, self.obj, self.lambda1, self.lambda2, opt_n_neurons)
