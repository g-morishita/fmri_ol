from pymer4.models import Lmer
import pandas as pd
from pathlib import Path


data_path = Path("../../../data/preprocessed").resolve()

def read_data():
    df = pd.read_csv(str(data_path / "preprocessed_data.csv"))
    return df


def fit(df):
    df["Y"] = df["self_choice"] == 2

    df["block"] = df["block"].astype("category")
    formula = "Y ~ 1 + is_partner_high_exp + block +  (1 + is_partner_high_exp + block | id)"
    model = Lmer(formula, data=df, family="binomial")
    model_fit = model.fit(control="optimizer='bobyqa', optCtrl = list(maxfun=5e5)")
    return model_fit

def main():
    df = read_data()
    model = fit(df)
    print(model)


if __name__ == "__main__":
    main()