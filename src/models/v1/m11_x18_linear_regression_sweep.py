import utils.core
from models.v1.m11_x18_linear_regression import M11_X18_LinearRegression

if __name__ == '__main__':
    config = {
        "lr": 1e-5,
        "weight_decay": 1e-5,
        "verbose": False,
        "batch_size": 256,
    }
    utils.core.train(M11_X18_LinearRegression, config=config, test=False, max_epochs=20, early_stopping_delta=1e-4)