from torch.optim.adam import Adam
from torch.optim.rmsprop import RMSprop
from torch.optim.adamw import AdamW

def build_optimizer(cfg, params):
    optimizer_type = cfg.get('type', 'adam')
    learning_rate = cfg.get('learning_rate', 0.001)
    weight_decay = cfg.get('weight_decay', 0)
    eps = float(cfg.get('eps', 1e-8))

    if optimizer_type == 'adam':
        momentum = cfg.get('momentum', 0.9)
        decay = cfg.get('decay', 0.999)
        betas = (momentum,decay)
        optimizer = Adam(params=params, lr=learning_rate, weight_decay=weight_decay, eps=eps, betas=betas)
    elif optimizer_type == 'rmsprop':
        momentum = cfg.get('momentum', 0)
        alpha = cfg.get('decay', 0.99)
        optimizer = RMSprop(params=params, lr=learning_rate, alpha=alpha, weight_decay=weight_decay, eps=eps,
                            momentum=momentum)
    elif optimizer_type == 'adamw':
        momentum = cfg.get('momentum', 0.9)
        decay = cfg.get('decay', 0.999)
        betas = (momentum,decay)
        optimizer = AdamW(params=params, lr=learning_rate, weight_decay=weight_decay, eps=eps, betas=betas)
    return optimizer
