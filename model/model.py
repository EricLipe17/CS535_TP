import os.path
import torch


class Padded3DCNN(torch.nn.Module):
    def __init__(self, num_classes, conv_layers, fc_layers, p_drop, loss_func=torch.nn.CrossEntropyLoss(),
                 save_fqp='./'):
        super(Padded3DCNN, self).__init__()
        self.num_classes = num_classes
        self.num_conv_layers = len(conv_layers)
        self.num_fc_layers = len(fc_layers)
        self.loss_func = loss_func
        self.optim = None
        self.scheduler = None
        self.save_fqp = os.path.abspath(save_fqp)

        if not os.path.exists(self.save_fqp):
            os.makedirs(self.save_fqp)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.layers = torch.nn.ParameterList()
        for conv_layer in conv_layers:
            self.layers.append(self._build_conv_layer(conv_layer[0], conv_layer[1], conv_layer[2], conv_layer[3],
                                                      conv_layer[4], conv_layer[5]))

        # TODO: Figure out how to calculate this from parameters
        prev_layer_size = 6400  # 2**3*conv_layers[-1][1]
        for fc_layer in fc_layers:
            self.layers.append(torch.nn.Linear(prev_layer_size, fc_layer))
            self.layers.append(torch.nn.LeakyReLU())
            prev_layer_size = fc_layer

        self.layers.append(torch.nn.InstanceNorm1d(fc_layers[-1]))

        self.layers.append(torch.nn.Dropout(p_drop))

        self.layers.append(torch.nn.Softmax(dim=1))

        # send everything to device
        self.layers.to(self.device)
        for layer in self.layers:
            layer.to(self.device)
        self.loss_func.to(self.device)
        self.to(self.device)

    def set_optimizer(self, optim):
        self.optim = optim

    @staticmethod
    def _build_conv_layer(in_channels, out_channels, conv_kernel_size, conv_stride, conv_padding, pool_kernel_size):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, conv_kernel_size, conv_stride, conv_padding),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(pool_kernel_size)
        )
        return layer

    def forward(self, frames):
        x = self.layers[0](frames)
        prev_was_conv = False
        for j in range(1, len(self.layers)):
            layer = self.layers[j]
            if type(layer) is torch.nn.Sequential:
                prev_was_conv = True
            if prev_was_conv and type(layer) is torch.nn.Linear:
                x = x.flatten()
                prev_was_conv = False
            if type(layer) is torch.nn.InstanceNorm1d:
                x = x.unsqueeze(0)
            x = layer(x)
        return x

    def train_model(self, train_loader, num_epochs, max_lr):
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=max_lr,
                                                             steps_per_epoch=int(len(train_loader)),
                                                             epochs=num_epochs,
                                                             anneal_strategy='linear')
        for i in range(num_epochs):
            for j, (label, frames) in enumerate(train_loader):
                # Move to device
                frames = frames.to(self.device)

                # Clear gradients
                self.optim.zero_grad()

                # Forward propagation
                prediction = self(frames)

                # Calculate softmax and cross entropy loss
                label = label.to(self.device)
                prediction = prediction.to(self.device)
                loss = self.loss_func(prediction, label)

                # Calculating gradients
                loss.backward()

                # Update parameters
                self.optim.step()
                self.scheduler.step()

                if j % 500 == 0 or j == 0 or j == len(train_loader) - 1:
                    print(f'Epoch: {i}, Iteration: {j}, Loss: {loss.data.item()}')
                    torch.save(self.state_dict(), os.path.join(self.save_fqp, 'model.pt'))


# TODO: Evaluate the data to make the inputs of the layers/model make sense
class Segmented3DCNN(torch.nn.Module):
    def __init__(self, num_classes, conv_layers, fc_layers, p_drop, loss_func=torch.nn.CrossEntropyLoss(),
                 save_fqp='./'):
        super(Segmented3DCNN, self).__init__()
        self.num_classes = num_classes
        self.num_conv_layers = len(conv_layers)
        self.num_fc_layers = len(fc_layers)
        self.loss_func = loss_func
        self.optim = None
        self.save_fqp = os.path.abspath(save_fqp)

        if not os.path.exists(self.save_fqp):
            os.makedirs(self.save_fqp)

        self.losses = list()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.layers = torch.nn.ParameterList()
        for conv_layer in conv_layers:
            self.layers.append(self._build_conv_layer(conv_layer[0], conv_layer[1], conv_layer[2], conv_layer[3],
                                                      conv_layer[4], conv_layer[5]))

        # TODO: Figure out how to calculate this from parameters
        prev_layer_size = 2304  # 2**3*conv_layers[-1][1]
        for fc_layer in fc_layers:
            self.layers.append(torch.nn.Linear(prev_layer_size, fc_layer))
            self.layers.append(torch.nn.LeakyReLU())
            prev_layer_size = fc_layer

        self.layers.append(torch.nn.Linear(fc_layers[-1], self.num_classes))

        self.layers.append(torch.nn.InstanceNorm1d(fc_layers[-1]))

        self.layers.append(torch.nn.Dropout(p_drop))

        self.layers.append(torch.nn.Softmax())

        for layer in self.layers:
            layer.to(self.device)

        # self.layers = [
        #     torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
        #     torch.nn.MaxPool3d(kernel_size=1, stride=1),
        #     torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
        #     torch.nn.MaxPool3d(kernel_size=1, stride=1),
        #     torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
        #     torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
        #     torch.nn.MaxPool3d(kernel_size=1, stride=1),
        #     torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
        #     torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
        #     torch.nn.MaxPool3d(kernel_size=1, stride=1),
        #     torch.nn.BatchNorm2d(num_features=1, device=None),
        #     torch.nn.LSTM(input_size=1, hidden_size=1, num_layers=1, bidirectional=True),
        #     torch.nn.Dropout(p=0.5),
        #     torch.nn.Linear(in_features=1024, out_features=512, device=None),
        #     torch.nn.Linear(in_features=512, out_features=256, device=None),
        #     torch.nn.Softmax(),
        # ]

    def set_optimizer(self, optim):
        self.optim = optim

    @staticmethod
    def _build_conv_layer(in_channels, out_channels, conv_kernel_size, conv_stride, conv_padding, pool_kernel_size):
        layer = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, conv_kernel_size, conv_stride, conv_padding),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool3d(pool_kernel_size)
        )
        return layer

    def forward(self, segments):
        forward_segments = torch.zeros((segments.shape[1], self.num_classes))
        for i in range(segments.shape[1]):
            segment = segments[0][i]
            x = self.layers[0](segment)
            prev_was_conv = False
            for j in range(1, len(self.layers)):
                layer = self.layers[j]
                if type(layer) is torch.nn.Sequential:
                    prev_was_conv = True
                if prev_was_conv and type(layer) is torch.nn.Linear:
                    x = x.flatten()
                    prev_was_conv = False
                if type(layer) is torch.nn.InstanceNorm1d:
                    x = x.unsqueeze(0)
                x = layer(x)
            forward_segments[i] = x.flatten()
        x = torch.mean(forward_segments, 0)
        return x

    def train_model(self, train_loader, num_epochs):
        for i in range(num_epochs):
            for j, (label, segments) in enumerate(train_loader):
                # Move to device
                segments = segments.to(self.device)

                # Clear gradients
                self.optim.zero_grad()

                # Forward propagation
                prediction = self(segments)

                # Calculate softmax and ross entropy loss
                label = label.reshape((label.shape[1],))
                label = label.to(self.device)
                prediction = prediction.to(self.device)
                loss = self.loss_func(prediction, label)

                # Calculating gradients
                loss.backward()

                # Update parameters
                self.optim.step()

                if j % 500 == 0 or j == 0 or j == len(train_loader) - 1:
                    print(f'Epoch: {i}, Iteration: {j}, Loss: {loss.data.item()}')
                    torch.save(self.state_dict(), os.path.join(self.save_fqp, 'model.pt'))

                # del segments, label, prediction, loss
