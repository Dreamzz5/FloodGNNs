import torch
import numpy as np


def masked_mae(
    prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, **args
) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Error (MAE) between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function is particularly useful for scenarios where the dataset contains missing or irrelevant
    values (denoted by `null_val`) that should not contribute to the loss calculation. It effectively
    masks these values to ensure they do not skew the error metrics.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor.
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor.
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A scalar tensor representing the masked mean absolute error.

    """

    target = target.repeat(len(prediction) // len(target), *([1] * (len(target.shape) - 1)))
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(
            target,
            torch.tensor(null_val).expand_as(target).to(target.device),
            atol=eps,
            rtol=0.0,
        )

    mask = mask.float()
    mask /= torch.mean(
        mask
    )  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero
    
    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    return torch.mean(loss)


# def interestingness_score(x, mean, std):
#     assert x.min() >= 0.0
#     comparable_discharge = (x.squeeze(-1).transpose(1, 2) / mean.to(x).unsqueeze(0)).flatten(0, 1)
#     mean_central_diff = torch.gradient(comparable_discharge, dim=-1)[0].mean()
#     trapezoid_integral = torch.trapezoid(comparable_discharge, dim=-1)

#     score = 1e3 * (mean_central_diff ** 2) * trapezoid_integral
#     assert not trapezoid_integral.isinf().any()
#     assert not trapezoid_integral.isnan().any()
#     return score.unsqueeze(-1)

# def masked_mae(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, **args) -> torch.Tensor:
#     """Masked Nash-Sutcliffe Efficiency.
#     Args:
#         prediction (torch.Tensor): predicted values
#         target (torch.Tensor): labels
#         null_val (float, optional): null value. Defaults to np.nan.
#     Returns:
#         torch.Tensor: masked Nash-Sutcliffe Efficiency
#     """
#     if np.isnan(null_val):
#         mask = ~torch.isnan(target)
#     else:
#         eps = 5e-5
#         mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.)
#     print(1)
#     mask = mask.float()
#     masked_target = target.masked_fill(~mask.bool(), 0)
#     masked_prediction = prediction.masked_fill(~mask.bool(), 0)

#     mean_target = args['mean'][..., 0:1].to(target).expand_as(target)
#     score = interestingness_score(args['inputs'], args['mean'][..., 0:1], args['std'])
#     if len(target.shape) == 4:
#         score = score.reshape(target.size(0), 1, -1, 1)
#     else:
#         score = score.reshape(target.size(0), -1, 1)
#     numerator = torch.sum(((masked_prediction - masked_target) * mask) ** 2 * score)
#     denominator = torch.sum(((masked_target - mean_target) * mask) ** 2 * score)

#     nse = 1 - (numerator / denominator)
#     return nse
