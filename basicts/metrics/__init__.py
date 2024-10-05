from .mae import masked_mae
from .mse import masked_mse, masked_nse, masked_huber
from .rmse import masked_rmse
from .mape import masked_mape
from .wape import masked_wape

ALL_METRICS = {
    "MAE": masked_mae,
    "MSE": masked_mse,
    "RMSE": masked_rmse,
    "MAPE": masked_mape,
    "WAPE": masked_wape,
    "NSE": masked_nse,
}

__all__ = [
    "masked_mae",
    "masked_mse",
    "masked_rmse",
    "masked_mape",
    "masked_wape",
    "masked_nse",
    "masked_huber",
    "ALL_METRICS",
]
