import torch
from torch import nn


class MLP_Block(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout):
        super(MLP_Block, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_features)
        self.Gelu = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.Gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Token_Mixer(nn.Module):
    def __init__(self, num_patches, num_channels, hidden_dim, dropout):
        super(Token_Mixer, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp_block = MLP_Block(num_patches, hidden_dim, dropout)

    def forward(self, x):
        initial = x
        x = self.layer_norm(x)
        x = torch.transpose(x, 1, 2)
        x = self.mlp_block(x)
        x = torch.transpose(x, 1, 2)
        output = initial + x
        return output


class Channel_Mixer(nn.Module):
    def __init__(self, num_channels, hidden_dim, dropout):
        super(Channel_Mixer, self).__init__()
        self.layer_norm = nn.LayerNorm(num_channels)
        self.mlp_block = MLP_Block(num_channels, hidden_dim, dropout)

    def forward(self, x):
        initial = x
        x = self.layer_norm(x)
        x = self.mlp_block(x)
        output = initial + x
        return output


class Mixer_Layer(nn.Module):
    def __init__(self, num_patches, num_channels, hidden_dim_token, hidden_dim_channel, dropout):
        super(Mixer_Layer, self).__init__()

        self.token_mixer = Token_Mixer(num_patches, num_channels, hidden_dim_token, dropout)
        self.channel_mixer = Channel_Mixer(num_channels, hidden_dim_channel, dropout)

    def forward(self, x):
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        return x


class MlpMixer(nn.Module):
    def __init__(self, image_shape: tuple,
                 patch_size: int,
                 num_classes,
                 num_mixers,
                 num_features,
                 hidden_dim_token,
                 hidden_dim_channel,
                 dropout=0.5):
        super(MlpMixer, self).__init__()

        in_channel = image_shape[0]

        assert image_shape[1] == image_shape[2], "Image must be square."
        assert image_shape[1] % patch_size == 0 and image_shape[2] % patch_size == 0, \
            "Image shape must be divisible by patch size"
        num_patches = (image_shape[1] // patch_size) ** 2

        # this conv layer is only for breaking the image into patches of latent dim size
        self.patch_breaker = nn.Conv2d(in_channel, num_features, kernel_size=patch_size, stride=patch_size)

        layers = []
        for _ in range(num_mixers):
            layers.append(Mixer_Layer(num_patches,
                                      num_features,
                                      hidden_dim_token,
                                      hidden_dim_channel,
                                      dropout))

        self.mixer_layers = nn.Sequential(*layers)

        self.final_fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        patches = self.patch_breaker(x)
        batch_size, num_features, h, w = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)

        patches = self.mixer_layers(patches)

        outputs = torch.mean(patches, dim=1)
        outputs = self.final_fc(outputs)

        return outputs


if __name__ == "__main__":
    model = MlpMixer(
        image_shape=(3, 32, 32),
        patch_size=4,
        num_classes=10,
        num_mixers=1,
        num_features=512,
        hidden_dim_token=256,
        hidden_dim_channel=128
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    img = torch.randn((1, 3, 32, 32), dtype=torch.float32, device=device)

    preds = model(img)
    print(preds)
