from fastapi import FastAPI
from pydantic import BaseModel
import torch
from torch import nn

app = FastAPI()


# Pydantic model with all 24 features
class CKDFeatures(BaseModel):
    age: float
    bp: float
    sg: float
    al: float
    su: float
    rbc: int
    pc: int
    pcc: int
    ba: int
    bgr: float
    bu: float
    sc: float
    sod: float
    pot: float
    hemo: float
    pcv: float
    wc: float
    rc: float
    htn: int
    dm: int
    cad: int
    appet: int
    pe: int
    ane: int


# PyTorch model
class Chronic_Kidney_Disease(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 28),
            nn.ReLU(),
            nn.Linear(28, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, X):
        return self.model(X)


INPUT_DIM = 24  # updated to match number of features
model = Chronic_Kidney_Disease(INPUT_DIM)
model.load_state_dict(torch.load("ckd_model.pth", map_location="cpu"))
model.eval()


# Prediction endpoint
@app.post("/predict")
def predict(features: CKDFeatures):
    x = torch.tensor(
        [
            [
                features.age,
                features.bp,
                features.sg,
                features.al,
                features.su,
                features.rbc,
                features.pc,
                features.pcc,
                features.ba,
                features.bgr,
                features.bu,
                features.sc,
                features.sod,
                features.pot,
                features.hemo,
                features.pcv,
                features.wc,
                features.rc,
                features.htn,
                features.dm,
                features.cad,
                features.appet,
                features.pe,
                features.ane,
            ]
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(x)
        prob = torch.sigmoid(logits).item()
        pred = "ckd" if prob > 0.5 else "notckd"
    return {"prediction": pred, "probability": prob}
