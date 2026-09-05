"""
Abstract base class for Wasserstein GAN with Gradient Penalty (WGAN-GP) training via PyTorch Lightning.
Implements the shared training loop, gradient penalty computation, and optimizer scheduling
used by the continuous and secondary generator stages.
"""

import torch
import logging

from abc import ABC, abstractmethod
from typing import OrderedDict
from pytorch_lightning import LightningModule
from torch.optim import RMSprop, Optimizer

import common.config as config


logging.basicConfig(level=logging.INFO)


class WGANInterface(LightningModule, ABC):
    """
    Abstract Base Class for Wasserstein GAN with Gradient Penalty (WGAN-GP).
    """
    def __init__(
            self,
            generator:       torch.nn.Module,
            critic:          torch.nn.Module,
            lr:              float,
            lambda_gp:       float,
            opt_g_frequency: int = 1,
            opt_d_frequency: int = 3,
    ):
        super().__init__()
        self.lr = lr
        self.lambda_gp = lambda_gp
        self.generator = generator
        self.critic = critic
        self.opt_g_frequency = opt_g_frequency
        self.opt_d_frequency = opt_d_frequency

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.generator(labels)

    @property
    def automatic_optimization(self) -> bool:
        return False

    def compute_gradient_penalty(
            self,
            labels: torch.Tensor,
            real_samples: torch.Tensor,
            fake_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the gradient penalty loss for the WGAN-GP.
        """
        batch_size = real_samples.size(0)
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((batch_size, 1)).to(self.device)

        # Determine the number of dimensions to unsqueeze alpha for broadcasting
        # real_samples shape might be (N, D) or (N, C, H, W) etc.
        # Here we assume (N, ...) so (N, 1) broadcast usually works for (N, D).
        # Adjust alpha shape to match real_samples except the first dimension
        while alpha.ndim > real_samples.ndim:
            real_samples = real_samples.unsqueeze(-1)

        while alpha.ndim > fake_samples.ndim:
            fake_samples = fake_samples.unsqueeze(-1)

        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates = interpolates.to(self.device)

        d_interpolates = self.critic(interpolates, labels)
        fake = torch.ones_like(d_interpolates).to(self.device)

        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Flatten gradients to (batch_size, -1)
        gradients = gradients.view(batch_size, -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def training_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> OrderedDict:
        self.log("learning rate", self.lr, on_epoch=True, on_step=False)
        labels, real_steps = batch

        opt_g, opt_d = self.optimizers()

        # Train Generator
        if batch_idx % (self.opt_g_frequency + self.opt_d_frequency) < self.opt_g_frequency:
             return self._train_generator_step(labels, opt_g)

        # Train Critic
        else:
            return self._train_critic_step(labels, real_steps, opt_d)

    def _train_generator_step(self, labels: torch.Tensor, optimizer: Optimizer) -> OrderedDict:
        generated_steps = self(labels)
        g_loss = -torch.mean(self.critic(generated_steps, labels))

        # Hook for additional loss terms (e.g. physics regularization)
        g_loss = self.add_generator_loss_terms(g_loss, labels, generated_steps)

        self.log("g_loss", g_loss, on_epoch=True, on_step=False)

        tqdm_dict = {'g_loss': g_loss}
        output = OrderedDict({
            'loss': g_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        if not self.automatic_optimization:
            optimizer.zero_grad()
            self.manual_backward(g_loss)
            # Clip gradients of the generator
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            optimizer.step()

        return output

    def _train_critic_step(self, labels: torch.Tensor, real_steps: torch.Tensor, optimizer: Optimizer) -> OrderedDict:
        fake_steps = self(labels)

        # Real steps
        real_validity = self.critic(real_steps, labels)
        # Fake steps
        fake_validity = self.critic(fake_steps, labels)

        # Gradient penalty
        gradient_penalty = self.compute_gradient_penalty(labels, real_steps, fake_steps)
        self.log("gradient_penalty", gradient_penalty, on_epoch=True, on_step=False)

        # Adversarial loss
        wgan_loss = torch.mean(real_validity) - torch.mean(fake_validity)
        self.log("Wasserstein distance estimate", wgan_loss, on_epoch=True, on_step=False)

        d_loss = -wgan_loss + self.lambda_gp * gradient_penalty
        self.log("critic_loss", d_loss, on_epoch=True, on_step=False)

        tqdm_dict = {'d_loss': d_loss}
        output = OrderedDict({
            'loss': d_loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        if not self.automatic_optimization:
            optimizer.zero_grad()
            self.manual_backward(d_loss)
            optimizer.step()

        return output

    def add_generator_loss_terms(self, g_loss: torch.Tensor, labels: torch.Tensor, generated_steps: torch.Tensor) -> torch.Tensor:
        """
        Override this method in subclasses to add extra loss terms (e.g. physics regularization).
        """
        return g_loss

    def configure_optimizers(self):
        max_lr = self.lr

        opt_g = RMSprop(self.generator.parameters(), lr=max_lr)
        opt_d = RMSprop(self.critic.parameters(), lr=max_lr)

        return (
            {
                'optimizer': opt_g,
                'frequency': self.opt_g_frequency
            },
            {
                'optimizer': opt_d,
                'frequency': self.opt_d_frequency
            }
        )

    def on_train_start(self) -> None:
        if not config.RESUME_FROM_CHECKPOINT:
            return None
        logging.info(f"Resuming from checkpoint: {self.current_epoch}")

        for optimizer in self.optimizers():
            for param_group in optimizer.param_groups:
                if param_group['lr'] != self.lr:
                    logging.info(f"Changed lr from {param_group['lr']} to {self.lr}")
                    param_group['lr'] = self.lr
        return None

    @abstractmethod
    def on_train_epoch_end(self):
        pass
