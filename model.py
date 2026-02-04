from cleaning import clean_ckd_dataset
import torch
from torch import nn
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42

df = clean_ckd_dataset()


X = df.drop("classification", axis=1).values
y = df["classification"].values
print(np.unique(X))  # make sure no strings remain

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)


class Chronic_Kidney_Disease(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=25, out_features=28),
            nn.ReLU(),
            nn.Linear(in_features=28, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, X) -> torch.Tensor:
        return self.model(X)


model_8 = Chronic_Kidney_Disease()


print(model_8.state_dict())


###  Settting up LossFunction & Optimizer   ###

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(params=model_8.parameters(), lr=0.05)


### Training L00P ####

epochs = 400
train_loss_list = []
test_loss_list = []
for epoch in range(epochs):
    model_8.train()
    y_logits = torch.sigmoid(model_8(X_train))
    y_preds = torch.round(y_logits)
    loss = loss_fn(y_logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_8.eval()
    with torch.inference_mode():
        test_logits = torch.sigmoid(model_8(X_test))
        test_loss = loss_fn(test_logits, y_test)
        test_preds = torch.round(test_logits)
        if epoch % 40 == 0:
            print(f"Epoch: {epoch}, loss: {loss:.2f}, test_loss: {test_loss:.2f}")
            train_loss_list.append(loss)
            test_loss_list.append(test_loss)
plt.title("🩺 Chronic Kidney Disease (CKD) Prediction System")
plt.plot(train_loss_list, test_loss_list)
plt.show()
