import torch

# TODO: Evaluate the data to make the inputs of the layers/model make sense
class CNN3D_LSTM(torch.nn.Module):
    def __int__(self):
        super(CNN3D_LSTM, self).__init()

        self.layers = [
            torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
            torch.nn.MaxPool3d(kernel_size=1, stride=1),
            torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
            torch.nn.MaxPool3d(kernel_size=1, stride=1),
            torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
            torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
            torch.nn.MaxPool3d(kernel_size=1, stride=1),
            torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
            torch.nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, device=None),
            torch.nn.MaxPool3d(kernel_size=1, stride=1),
            torch.nn.BatchNorm2d(num_features=1, device=None),
            torch.nn.LSTM(input_size=1, hidden_size=1, num_layers=1, bidirectional=True),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features=1024, out_features=512, device=None),
            torch.nn.Linear(in_features=512, out_features=256, device=None),
            torch.nn.Softmax(),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x