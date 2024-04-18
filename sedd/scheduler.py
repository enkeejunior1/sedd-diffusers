import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Union

class ScoreEntropyLoss(nn.Module):
    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        pass

    def forward(self, log_score, t, xt, x0, reduction='sum'):
        '''
        TODO: need to verify the code
        '''
        sigma_bar = self.scheduler.sigma_bar[t].unsqueeze(1).expand_as(xt)
        sigma = self.scheduler.sigma[t].unsqueeze(1).expand_as(xt)

        perturbed_pos = xt == self.scheduler.num_vocabs - 1
        ratio = 1 / sigma_bar[perturbed_pos].expm1() # p(y|x0) / p(xt|x0) = exp(-sigma) / (1 - exp(-sigma))
        y = x0[perturbed_pos]

        pos = log_score[perturbed_pos][:, :-1].exp().sum(dim=-1) # pos = torch.gather(log_score[perturbed_pos].exp(), -1, y[..., None]).squeeze()
        neg = ratio * torch.gather(log_score[perturbed_pos], -1, y[..., None]).squeeze()
        const = ratio * (ratio.log() - 1) # there are no constant term in algorithm 1

        loss = sigma[perturbed_pos] * (pos - neg + const) # DWDSE loss
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()
    

class Scheduler:
    '''
    Train 
        1. t, samples -> sigma (alphas_comprod)  - (sample_transition) -> noisy_samples
        2. pred_score = model(samples, t)
        3. score = get_score(samples, noisy_samples)
        4. loss_weight = get_loss_weight(t)
        5. loss = loss_weight * comp_loss(pred_score, score)
        
    Sampling
    '''
    def __init__(
        self, config
    ):  
        # basic configs
        if config.graph.type == 'uniform':
            self.num_vocabs = config.dataset.tokens 
        if config.graph.type == 'absorb':
            self.num_vocabs = config.dataset.tokens + 1
        
        # init timesteps
        self.num_train_timesteps = config.noise.num_train_timesteps
        # self.timesteps = torch.from_numpy(
        #     np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        # )
        
        # init noise schedule (similar to alphas_cumprod)
        self.get_sigma_bar(config.noise) # \int_0^t exp(sigma(t)) dt
        
        # init graph
        # self.graph = self.get_graph(config.graph)
        
    def to(self, device, dtype):
        self.sigma_bar = self.sigma_bar.to(device, dtype)   
        self.sigma = self.sigma.to(device, dtype)   
    
    def get_sigma_bar(
        self, config_noise = None,
    ):
        t = np.linspace(0, 1, self.num_train_timesteps)
        
        if config_noise.type == 'loglinear':
            self.sigma_bar = torch.from_numpy(
                -np.log(1 - (1 - config_noise.eps) * t)
            )
            self.sigma = torch.from_numpy(
                (1-config_noise.eps) / (1 - (1-config_noise.eps) * t)
            )
                
        elif config_noise.type == 'geometric':
            self.sigma_bar = torch.from_numpy(
                config_noise.sigma_min ** (1-t) * config_noise.sigma_max ** t
            )
            raise NotImplementedError('')
            
        else:
            raise ValueError(f'please check noise.type: {config_noise.type}')
    
    def add_noise(
        self, samples: torch.FloatTensor, t: Union[int, torch.LongTensor],
    ):
        # snr
        sigma_bar = self.sigma_bar[t]
        
        # perturb samples (absorb)
        perturb_prob = 1 - (-sigma_bar).exp()
        perturbed_samples = torch.where(
            torch.rand(*samples.shape, device=samples.device) < perturb_prob[:, None],
            self.num_vocabs - 1, samples
        )

        return perturbed_samples
    
    def set_timesteps(
        self, num_inference_steps: int = None, offset: int = 0, device: Union[str, torch.device] = None,
    ):  
        '''follow set_timesteps, leading'''
        step_ratio = self.num_train_timesteps // num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        timesteps += offset

        self.timesteps = torch.from_numpy(timesteps).to(device)


    # Tweedie sampler
    # def step(
    #     self, score: torch.FloatTensor, t: int, xt: torch.FloatTensor, 
    # ):
    #     import torch.nn.functional as F
    #     assert (score >= 0).all()
    #     '''TODO: reimplement the code w.r.t paper'''

    #     # get next timestep (t - dt)
    #     prev_t = self.timesteps[(self.timesteps == t).long().argmax()+1]

    #     sigma_bar_t = self.sigma_bar[t]
    #     sigma_bar_prev_t = self.sigma_bar[prev_t]

    #     # staggered_score
    #     stag_score = score.clone()
    #     extra_const = (1 - (sigma_bar_t).exp()) * stag_score.sum(dim=-1)
    #     stag_score *= sigma_bar_t.exp()[:, None]
    #     stag_score[..., -1] += extra_const

    #     # edge 
    #     sigma_bar_t = unsqueeze_as(sigma_bar_t, xt[..., None])
    #     edge = (-sigma_bar_t).exp() * F.one_hot(xt, num_classes=self.num_vocabs)
    #     edge += torch.where(
    #         xt == self.num_vocabs - 1,
    #         1 - (-sigma_bar_t).squeeze(-1).exp(),
    #         0
    #     )[..., None]

    #     # sample prob
    #     probs = stag_score * edge
    #     probs = probs[..., :-1]

    #     return sample_categorical(probs)
        
    # Euler sampler
    def step(
        self, score: torch.FloatTensor, t: Union[int, torch.LongTensor], xt: torch.FloatTensor, 
    ):
        import torch.nn.functional as F
        assert (score >= 0).all()
        '''TODO: reimplement the code w.r.t paper'''
        # step size
        prev_t = self.timesteps[(self.timesteps == t).long().argmax()+1]
        dt = (t - prev_t) / self.num_train_timesteps

        if isinstance(t, int):
            t = torch.tensor(t, device=xt.device).unsqueeze(0).repeat(xt.size(0))

        sigma = self.sigma[t]

        # normalized edge (transp_rate -> reverse_rate) # TODO: actually, didn't understand this. 
        edge = -F.one_hot(xt, num_classes=self.num_vocabs)
        edge[xt == self.num_vocabs - 1] += 1

        normalized_rate = edge * score
        normalized_rate.scatter_(-1, xt[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, xt[..., None], -normalized_rate.sum(dim=-1, keepdim=True))

        rev_rate = dt * sigma[..., None] * normalized_rate
        
        # reverse step
        probs = F.one_hot(xt, num_classes=self.num_vocabs).to(rev_rate) + rev_rate
        return sample_categorical(probs)


def sample_categorical(categorical_probs, method="hard"):
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    

def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)
