import random

import torch

def generalized_lambda_returns(rewards,
                               pcontinues,
                               state_values,
                               bootstrap_value,
                               lambda_=1, ):
    r"""
    Overview:
        Same as trfl.sequence_ops.multistep_forward_view
        ```python
        result[t] = rewards[t] + pcontinues[t]*(lambda_[t]*result[t+1] + (1-lambda_[t])*state_values[t+1])
        result[last] = rewards[last] + pcontinues[last]*state_values[last]
        ```
    Args:
        rewards: 2-D Tensor with shape `[T, B]`.
        pcontinues: 2-D Tensor with shape `[T, B]`.
        values: 2-D Tensor containing estimates of the state values for timesteps  0 to `T-1`. Shape `[T , B]`.
        bootstrap_value: 1-D Tensor containing an estimate of the value of the final state at time `T`,
                used for bootstrapping the target n-step returns. Shape `[B]`.
        lambda_: an optional scalar or 2-D Tensor with shape `[T, B]`.
    Returns:
        2-D Tensor with shape `[T, B]`
    """
    if not isinstance(lambda_, torch.Tensor):
        lambda_ = lambda_ * torch.ones_like(rewards)
    result = torch.empty_like(rewards)
    result[-1,] = rewards[-1,] + pcontinues[-1,] * bootstrap_value
    for t in reversed(range(rewards.size()[0] - 1)):
        result[t] = rewards[t,] + pcontinues[t] * (lambda_[t] * result[t + 1,] + (1 - lambda_[t]) * state_values[t + 1])
    return result


def td_lambda_loss(state_values, rewards, pcontinues, lambda_=0.8, ):
    r"""
    Overview:
        Computing TD($\lambda$) loss given constant gamma and lambda.
        There is no special handling for terminal state value,
        if some state has reached the terminal, just fill in zeros for values and rewards beyond terminal
        (*including the terminal state*, values[terminal] should also be 0)
    Arguments:
        - values (:obj:`torch.Tensor`): estimation of the state value at step 0 to T, of size [T + 1, B]
        - rewards (:obj:`torch.Tensor`): the returns from time step 0 to T-1, of size [T_traj, batchsize]
        - lambda_ (:obj:`float`): constant lambda (between 0 to 1)
    Returns:
        - loss (:obj:`torch.Tensor`): Computed MSE loss, averaged over the batch, of size []
    """
    with torch.no_grad():
        returns = generalized_lambda_returns(rewards=rewards,
                                             pcontinues=pcontinues,
                                             state_values=state_values[:-1],
                                             bootstrap_value=state_values[-1],
                                             lambda_=lambda_, )
    # discard the value at T as it should be considered in the next slice
    loss = 0.5 * torch.pow(returns - state_values[:-1], 2)
    loss = loss.mean()
    return loss

if __name__ == '__main__':
    unroll_len = 5
    batch_size = 3
    lambda_= random.random()
    rewards = torch.randn(size=(unroll_len,batch_size,),)
    dones = torch.randint(low=0,high=2,size=(unroll_len,batch_size,),dtype=torch.bool)
    pcontinues = 1 - dones.float()
    state_values = torch.randn(size=(unroll_len+1,batch_size,),)
    loss = td_lambda_loss(state_values, rewards, pcontinues, lambda_=lambda_, )
    print(loss)
