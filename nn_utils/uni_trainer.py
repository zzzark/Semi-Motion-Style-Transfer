import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import copy
from os.path import join as pj
import os


class GenModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def get_loss(self, *args) -> [torch.Tensor, dict]:
        """
        return loss, loss dict
        """
        pass


class GANModel(nn.Module, ABC):
    # TODO:
    #   split to `GenModel` and `DisModel`
    #   so `GenModel` can be reused.
    pass


class GenTrainer:
    def __init__(self, model: GenModel, device,
                 optim='Adam', param_lr=1e-4, param_wd=1e-4,
                 use_ema=False, parallel_device=None):

        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f'total params: {pytorch_total_params:,}')

        self.device = device
        self._model = model
        self._m_ema = copy.deepcopy(self._model)

        self._model = self._model.to(device)
        self._m_ema = self._m_ema.to(device)

        self.use_ema = use_ema

        if optim == 'RAdam':
            self.optim = torch.optim.RAdam(self._model.parameters(), lr=param_lr, weight_decay=param_wd)
        elif optim == 'Adam':
            self.optim = torch.optim.Adam(self._model.parameters(), lr=param_lr, weight_decay=param_wd)
        elif optim == 'RMSProp':
            self.optim = torch.optim.RMSprop(self._model.parameters(), lr=param_lr, weight_decay=param_wd)
        else:
            raise NotImplementedError(f"{optim}")

        if parallel_device is not None:
            self.parallel = nn.DataParallel(self._model, device_ids=parallel_device)
        else:
            self.parallel = None

    @property
    def model(self):
        return self._m_ema if self.use_ema else self._model

    @staticmethod
    def _update_average(model_tgt, model_src, beta=0.999):
        with torch.no_grad():
            param_dict_src = dict(model_src.named_parameters())
            for p_name, p_tgt in model_tgt.named_parameters():
                p_src = param_dict_src[p_name]
                assert (p_src is not p_tgt)
                p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    def update_once(self, *data) -> dict:
        exec_module = self._model if self.parallel is None else self.parallel

        data = [
            e.to(self.device) if isinstance(e, torch.Tensor) else e
            for e in data
        ]
        self.optim.zero_grad()

        output = exec_module(*data)
        loss_tot, loss_dic = self._model.get_loss(*output)
        loss_tot.backward()

        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self.optim.step()

        if self.use_ema:
            self._update_average(self._m_ema, self._model)

        return loss_dic

    def resume(self, model_dir, device, start_iter=None) -> int:
        import glob
        ls = [-1]  # fallback
        for file in glob.glob(pj(model_dir, "*.pt")):
            try:
                ls.append(int(file[-9:-3]))
            except ValueError:
                pass
        ls.sort()

        if start_iter is None:
            iterations = ls[-1]
        else:
            iterations = start_iter

        def __try_load(mdl, pth):
            if os.path.isfile(pth):
                mdl.load_state_dict(torch.load(pth, map_location=device))
            else:
                print(f'[WARNING] file not found {pth}')

        if iterations >= 0:
            ema_name = os.path.join(model_dir, 'ema_%06d.pt' % iterations)
            mdl_name = os.path.join(model_dir, 'mdl_%06d.pt' % iterations)
            opt_name = os.path.join(model_dir, 'optimizer.pt')
            if self.use_ema:
                __try_load(self._m_ema, ema_name)
            __try_load(self._model, mdl_name)
            __try_load(self.optim, opt_name)

        return iterations

    def save(self, model_dir, iterations):
        os.makedirs(model_dir, exist_ok=True)

        mdl_name = os.path.join(model_dir, 'mdl_%06d.pt' % iterations)
        ema_name = os.path.join(model_dir, 'ema_%06d.pt' % iterations)
        opt_name = os.path.join(model_dir, 'optimizer.pt')
        if self.use_ema:
            torch.save(self._m_ema.state_dict(), ema_name)
        torch.save(self._model.state_dict(), mdl_name)
        torch.save(self.optim.state_dict(), opt_name)


class GANTrainer:
    pass
