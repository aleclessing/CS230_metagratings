import jnet
import torch.device


def train_model(model, epochs, batch_size, learning_rate, device):
    pass


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = jnet.JNet(im_dim=(64, 256), static_channels=1, dynamic_channels=3)

    dataset = 


    train_model(
            model=model,
            epochs=10,
            batch_size=10,
            learning_rate=0.001,
            device=device)