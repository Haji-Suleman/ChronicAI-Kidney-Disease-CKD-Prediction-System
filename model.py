from cleaning import clean_ckd_dataset
import torch
from torch import nn
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

df = clean_ckd_dataset()


X = df.drop("classification", axis=1).values
y = df["classification"].values


X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


class Chronic_Kidney_Disease(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=26, out_features=28),
            nn.ReLU(),
            nn.Linear(in_features=28, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, X) -> torch.Tensor:
        return self.model(X)


model_8 = Chronic_Kidney_Disease()
