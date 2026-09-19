import torch
import trainer
from utils import ARGS, get_data_loader
from simple_cnn import SimpleCNN
from voc_dataset import VOCDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class ResNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()

        self.resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        ##################################################################
        # TODO: Define a FC layer here to process the features
        ##################################################################
        self.resnet.fc = nn.Linear(512, num_classes)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################
        

    def forward(self, x):
        ##################################################################
        # TODO: Return raw outputs here
        ##################################################################
        return self.resnet(x)
        ##################################################################
        #                          END OF YOUR CODE                      #
        ##################################################################


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    ##################################################################
    # TODO: Create hyperparameter argument class
    # Note that you might have to change the augmentations
    # You should experiment and choose the correct hyperparameters
    # Aim for an mAP of around 0.8 (80%) in 50 epochs.
    ##################################################################
    args = ARGS(
        epochs=1,
        inp_size=224,
        use_cuda=True,
        val_every=150,
        lr= 1e-4,
        batch_size=32,
        step_size=5,
        gamma=0.1,
    )
    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################
    
    print(args)

    ##################################################################
    # TODO: Define a ResNet-18 model (https://arxiv.org/pdf/1512.03385.pdf) 
    # Initialize this model with ImageNet pre-trained weights
    # (except the last layer). You are free to use torchvision.models 
    ##################################################################

    model = ResNet(len(VOCDataset.CLASS_NAMES)).to(args.device)

    ##################################################################
    #                          END OF YOUR CODE                      #
    ##################################################################

    # initializes Adam optimizer and simple StepLR scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # trains model using your training code and reports test map
    test_ap, test_map = trainer.train(args, model, optimizer, scheduler)
    print('test map:', test_map)

    # Q2 t-SNE plot guidance:
    # Use this fine-tuned Q2 classifier in evaluation mode, not a fresh
    # ImageNet model. Extract the average pooling output immediately before
    # the final fully connected layer (resnet.avgpool), and flatten it to
    # one 512-dimensional feature vector per image for ResNet-18.
    # Use the 20 base PASCAL VOC classes (VOCDataset.CLASS_NAMES) in the
    # legend, rather than creating entries for multi-label combinations.

    model.fc = torch.nn.Identity()
    model = model.eval()

    voc_data_loader = get_data_loader('voc', train=False, batch_size=1, split='test', inp_size=224)

    features = []
    labels = []
    num_samples = 0
    with torch.no_grad():
        for data, target, _ in voc_data_loader:
            if target.sum() != 1:
                continue
            data = data.to(args.device)
            f = model(data)

            features.append(f.cpu().numpy().squeeze(0))
            labels.append(target.argmax(dim=-1).item())
            num_samples += 1
            if num_samples >= 1000:
                break

    features = np.array(features)
    labels = np.array(labels)

    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(features)

    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=labels,
        cmap="tab20",
        alpha=0.8,
        edgecolors="none",
    )

    ax.set_title("2D Projection of ResNet Features", fontsize=14, pad=15)
    ax.grid(True, linestyle="--", alpha=0.5)

    handles, _ = scatter.legend_elements(num=20)
    ax.legend(
        handles,
        VOCDataset.CLASS_NAMES,
        title="Classes",
    )

    plt.savefig("tsne_plot.png", dpi=300, bbox_inches="tight")
    plt.close()