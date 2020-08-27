import torch
import librosa
import numpy as np


def get_fourier_weights_for_mel(window_size, hanning=True):
    frec = np.linspace(-window_size//2, 0, window_size//2+1)
    time = np.linspace(0, window_size-1, window_size)
    
    if hanning:
        hanning_window = np.hanning(window_size)

    filters_cos = []
    filters_sin = []
    for f in frec:
        filters_cos.append(np.cos(2*np.pi*f*time/window_size))
        filters_sin.append(np.sin(2*np.pi*f*time/window_size))
        
    if hanning:
        filters_cos = np.array(filters_cos)[::-1]*hanning_window
        filters_sin = np.array(filters_sin)[::-1]*hanning_window
    else:
        filters_cos = np.array(filters_cos)[::-1]
        filters_sin = np.array(filters_sin)[::-1]
    return filters_cos, filters_sin


class STFTModel(torch.nn.Module):
    def __init__(self, window_size = 2048, train_fourier=False, sr=22050, hop_size=128):
        super().__init__()
        
        kernel_size = window_size
        stride = hop_size
        filters = kernel_size//2 + 1
        
        self.cos = torch.nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size, stride=stride, bias=False)
        self.sin = torch.nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size, stride=stride, bias=False)

        cos_weights, sin_weights = get_fourier_weights_for_mel(window_size)
        self.cos.weight.data = torch.from_numpy(cos_weights.reshape(cos_weights.shape[0], 1, cos_weights.shape[1])).float()
        self.sin.weight.data = torch.from_numpy(sin_weights.reshape(sin_weights.shape[0], 1, sin_weights.shape[1])).float()
            
        list(self.cos.parameters())[0].requires_grad = train_fourier
        list(self.sin.parameters())[0].requires_grad = train_fourier

    def forward(self, x):
        stft = self.cos(x)**2 + self.sin(x)**2
        x_spec = 10.0 * torch.log10(stft)
        return x_spec
    
class MelSpectrogramModel(torch.nn.Module):
    def __init__(self, window_size = 2048, train_fourier=False, sr=22050, hop_size=128, n_mels=128, train_mel=False, fmin=0, fmax=None, in_db=False):
        super().__init__()
        self.in_db = in_db
        kernel_size = window_size
        stride = hop_size
        filters = kernel_size//2 + 1
        
        # STFT
        self.cos = torch.nn.Conv1d(1, filters, kernel_size, stride=stride, bias=False)
        self.sin = torch.nn.Conv1d(1, filters, kernel_size, stride=stride, bias=False)

        cos_weights, sin_weights = get_fourier_weights_for_mel(window_size)
        self.cos.weight.data = torch.from_numpy(cos_weights.reshape(cos_weights.shape[0], 1, cos_weights.shape[1])).float()
        self.sin.weight.data = torch.from_numpy(sin_weights.reshape(sin_weights.shape[0], 1, sin_weights.shape[1])).float()
        
        # No entrenables
        list(self.cos.parameters())[0].requires_grad = train_fourier
        list(self.sin.parameters())[0].requires_grad = train_fourier
        
        # MEL
        mel_filters = librosa.filters.mel(sr, n_fft=window_size, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.mel_filter = torch.nn.Conv1d(in_channels=mel_filters.shape[1], out_channels=mel_filters.shape[0], kernel_size=1, bias=False)
        self.mel_filter.weight.data[:,:,0] = torch.from_numpy(mel_filters)
        
        # No entrenable
        list(self.mel_filter.parameters())[0].requires_grad = train_mel

    def forward(self, x):
        stft = self.cos(x)**2 + self.sin(x)**2
        mel_out = self.mel_filter(stft)
        if self.in_db:
            mel_out_db = 10.0 * torch.log10(mel_out)
            return mel_out_db
        else:
            return mel_out
        
def resnet_BW(resnet):
    resnet.conv1.in_channels = 1
    conv1_shape = np.array(resnet.conv1.weight.data.shape)
    conv1_shape[1] = 1
    # Las componentes RGB son todas iguales, tomo factor comun y los pesos se suman
    resnet.conv1.weight.data = resnet.conv1.weight.data.sum(axis=1).reshape(*conv1_shape)
    return resnet
        
class BirdsNet(torch.nn.Module):
    def __init__(self, window_size=2048, hop_size=128, n_mels=128, sr=22050, pretrained=True, n_classes=264, resnet_type='resnet18', init_fourier=True, init_mel=True, train_fourier=False, train_mel=False, a=-1.2, fully_connected=False, fmin=0.0, fmax=None, dropout=0.2):
        super().__init__()
        
        
        self.mel_spectrogram = MelSpectrogramModel(
            window_size = window_size, train_fourier=train_fourier, sr=sr, hop_size=hop_size, n_mels=n_mels, train_mel=train_mel, fmin=fmin, fmax=fmax
        )

        self.a = torch.nn.Parameter(torch.tensor([a]))
        
        self.bn1 = torch.nn.BatchNorm2d(1)
        
        model_resnet = torch.hub.load('pytorch/vision:v0.6.0', resnet_type, pretrained=pretrained)
        model_resnet_BW = resnet_BW(model_resnet)
        
        self.resnet = torch.nn.Sequential(*list(model_resnet_BW.children())[:-1])
        
        if not fully_connected:
            self.conv_out = torch.nn.Conv2d(model_resnet_BW.fc.weight.shape[1], n_classes, 1)
        else:
            self.conv_out = torch.nn.Sequential(
                torch.nn.Conv2d(model_resnet_BW.fc.weight.shape[1], hidden_units, 1), torch.nn.ReLU(), torch.nn.Dropout(p=dropout),
                torch.nn.Conv2d(hidden_units, hidden_units, 1), torch.nn.ReLU(), torch.nn.Dropout(p=dropout),
                torch.nn.Conv2d(hidden_units, n_classes, 1)
            )
            
        
    
    def forward(self, x):
        mel_spectrogram = self.mel_spectrogram(x)
        mel_spectrogram = torch.pow(mel_spectrogram, torch.sigmoid(self.a))
        
        mel_spectrogram = mel_spectrogram.reshape(-1, 1, *mel_spectrogram.shape[1:])
        mel_spectrogram_normalized = self.bn1(mel_spectrogram)
        x = self.resnet(mel_spectrogram_normalized)
        if len(x.shape) == 2:
            x = x.reshape(*x.shape, 1, 1)
        x = self.conv_out(x).flatten(start_dim=1)
        return mel_spectrogram_normalized, x