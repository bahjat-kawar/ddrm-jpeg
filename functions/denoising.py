import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os

from functions.jpeg_torch import jpeg_decode as jd, jpeg_encode as je


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a 
    
def jpeg_steps(x, seq, model, b, y_0, etaB, etaA, etaC, cls_fn=None, classes=None, jpeg_qf=None):
    from functools import partial
    jpeg_decode = partial(jd, qf = jpeg_qf)
    jpeg_encode = partial(je, qf = jpeg_qf)
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        
        a_init = compute_alpha(b, (torch.ones(n) * seq[-1]).to(x.device).long())

        xs = [a_init.sqrt() * y_0 + (1 - a_init).sqrt() * torch.randn_like(x)]
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            if cls_fn == None:
                et = model(xt, t)
            else:
                et = model(xt, t, classes)
                et = et[:, :3]
                et = et - (1 - at).sqrt()[0,0,0,0] * cls_fn(x,t,classes)
            
            if et.size(1) == 6:
                et = et[:, :3]
            
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]

            xt_next = x0_t
            xt_next = xt_next - jpeg_decode(jpeg_encode(xt_next)) + jpeg_decode(jpeg_encode(y_0))

            xt_next = etaB * at_next.sqrt() * xt_next + (1 - etaB) * at_next.sqrt() * x0_t + etaA * (1 - at_next).sqrt() * torch.randn_like(xt_next) + (1 - etaA) * et * (1 - at_next).sqrt()

            x0_preds.append(x0_t.to('cpu'))
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds