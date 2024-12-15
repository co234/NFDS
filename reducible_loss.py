import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

import torch.nn as nn
from torch.nn import functional as F
from utils import top_x_indices, create_logging_dict


# This is adpated from RHO-LOSS 
# https://github.com/OATML/RHO-Loss

class reducible_loss_selection:
    def __call__(
        self,
        selected_batch_size,
        data=None,
        target=None,
        sensitive_attrs = None,
        global_index=None,
        large_model=None,
        irreducible_loss_generator=None,
        proxy_model=None,
        l = 0.5,
        *args,
        **kwargs,
    ):
 
        with torch.no_grad():
            if proxy_model is not None:
                model_loss = F.cross_entropy(
                    proxy_model(data), target, reduction="none"
                )
            else:
                model_loss = F.cross_entropy(
                                    large_model(data), target, reduction="none"
                )
      
            irreducible_loss = _compute_irreducible_loss(
                data,
                target,
                global_index,
                irreducible_loss_generator['irreducible_losses']
            )

            # =========== Add fair regularizaers ==========
            a_idx,b_idx = np.where(sensitive_attrs==1)[0], np.where(sensitive_attrs==0)[0]
            reducible_loss = model_loss - irreducible_loss

            loss_category = {}
            loss_category[1] = reducible_loss[a_idx]
            loss_category[0] = reducible_loss[b_idx]
            
            #==========================================

            sorted_a_idx = torch.argsort(loss_category[1], descending=True)
            sorted_b_idx = torch.argsort(loss_category[0], descending=True)
            top_x_idx_a, top_x_idx_b = sorted_a_idx[:int(selected_batch_size/2)], sorted_b_idx[:int(selected_batch_size/2)]
        
            selected_minibatch = torch.cat((top_x_idx_a,top_x_idx_b))


        return selected_minibatch




def _compute_irreducible_loss(
    data=None,
    target=None,
    global_index=None,
    irreducible_loss_generator=None,
):
    if type(irreducible_loss_generator) is torch.Tensor:
        # send the whole tensor over
        irreducible_loss = irreducible_loss_generator[global_index]
    else:
        irreducible_loss = F.cross_entropy(
            irreducible_loss_generator(data), target, reduction="none"
        )

    return irreducible_loss
