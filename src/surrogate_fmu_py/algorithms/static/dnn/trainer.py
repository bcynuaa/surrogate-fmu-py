from __future__ import annotations

from typing import List, Dict, Callable
from copy import deepcopy
from dataclasses import dataclass, asdict, field

import torch

from ....contracts import TrainerLike, CoderLike, ReporterLike
from ....arrays import Tensor, to_torch
from ....implements import NoneReporter
from ....utils import get_loss_function, create_optimizer
from ..implements import StaticTorchDataset
from .core import DNNCore

@dataclass(frozen=True)
class DNNTrainerConfig:
    seed: int = 42
    batch_size: int = 32
    shuffle: bool = True
    optimizer: str = "adam"
    loss_type: str = "mse"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    learning_rate_decay: float = 0.99
    epochs: int = 300
    early_stop: bool = True
    patience: int = 20
    min_delta: float = 0.0
    device: str = "cpu"
    channel_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    @classmethod
    def from_dict(cls, config: dict) -> DNNTrainerConfig:
        seed: int = config.get("seed", 42)
        batch_size: int = config.get("batch_size", 32)
        shuffle: bool = config.get("shuffle", True)
        optimizer: str = config.get("optimizer", "adam")
        loss_type: str = config.get("loss_type", "mse")
        learning_rate: float = config.get("learning_rate", 0.001)
        weight_decay: float = config.get("weight_decay", 0.0)
        learning_rate_decay: float = config.get("learning_rate_decay", 0.99)
        epochs: int = config.get("epochs", 300)
        early_stop: bool = config.get("early_stop", True)
        patience: int = config.get("patience", 20)
        min_delta: float = config.get("min_delta", 0.0)
        device: str = config.get("device", "cpu")
        channel_weights: List[float] = config.get("channel_weights", [1.0, 1.0, 1.0])
        return cls(
            seed=seed,
            batch_size=batch_size,
            shuffle=shuffle,
            optimizer=optimizer,
            loss_type=loss_type,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
            weight_decay=weight_decay,
            epochs=epochs,
            early_stop=early_stop,
            patience=patience,
            min_delta=min_delta,
            device=device,
            channel_weights=channel_weights
        )
        pass

    def to_dict(self) -> Dict:
        return asdict(self)
        pass

    pass


class DNNTrainer(torch.nn.Module, TrainerLike):
    
    def __init__(self) -> None:
        super().__init__()
        self.config_: DNNTrainerConfig = DNNTrainerConfig()
        pass

    def init(self, params: Dict) -> DNNTrainer:
        self.config_ = DNNTrainerConfig.from_dict(config=params)
        return self
        pass

    def train(
        self,
        coder: CoderLike,
        core: DNNCore,
        train_dataset: StaticTorchDataset,
        validation_dataset: StaticTorchDataset,
        reporter: ReporterLike | None = None
    ) -> DNNCore:
        if reporter is None:
            reporter = NoneReporter()
            pass
        reporter.message("DNN: Start Training, Prepare Components.")
        reporter.progress(0.0, message="DNN: Preparing Training Components.")
        torch.manual_seed(self.config_.seed)
        train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.config_.batch_size,
            shuffle=self.config_.shuffle,
        )
        validation_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset=validation_dataset,
            batch_size=self.config_.batch_size,
            shuffle=False,
        )
        device: torch.device = torch.device(self.config_.device)
        coder.to_torch()
        core.to(device)
        optimizer: torch.optim.Optimizer = create_optimizer(
            optimizer_name=self.config_.optimizer,
            params=core.parameters(),
            learning_rate=self.config_.learning_rate,
            weight_decay=self.config_.weight_decay,
        )
        scheduler: torch.optim.lr_scheduler.ExponentialLR = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=self.config_.learning_rate_decay
        )
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = get_loss_function(loss_type=self.config_.loss_type)
        best_loss: float = float("inf")
        best_state_dict = deepcopy(core.state_dict())
        epochs_without_improvement: int = 0
        epochs: int = self.config_.epochs
        # * assuming channel_weights have been formatted from top level's algorithm.format_params(overrides)
        channel_weights_tensor: Tensor = torch.tensor(self.config_.channel_weights, dtype=torch.float32, device=device)
        reporter.message(f"DNN: Preparation Done. Start Training for {epochs} epochs.")
        reporter.progress(5.0, message="DNN: Start Training, Enter Epochs.")
        for epoch in range(epochs):
            core.train()
            train_loss: float = 0.0
            batch_count: int = 0
            for batch in train_loader:
                x_batch, y_batch = batch
                xb: Tensor = to_torch(coder.encode_input(x_batch)).to(device)
                yb: Tensor = to_torch(coder.encode_output(y_batch)).to(device)
                optimizer.zero_grad()
                y_pred: Tensor = core(xb)
                loss: Tensor = loss_fn(y_pred * channel_weights_tensor, yb * channel_weights_tensor)
                loss.backward()
                optimizer.step()
                batch_count += x_batch.shape[0]
                train_loss += loss.cpu().item() * x_batch.shape[0]
                pass
            train_loss /= batch_count
            validation_loss: float = 0.0
            core.eval()
            with torch.no_grad():
                batch_count = 0
                for batch in validation_loader:
                    x_batch, y_batch = batch
                    xb = to_torch(coder.encode_input(x_batch)).to(device)
                    yb = to_torch(coder.encode_output(y_batch)).to(device)
                    y_pred = core(xb)
                    loss = loss_fn(y_pred * channel_weights_tensor, yb * channel_weights_tensor)
                    batch_count += x_batch.shape[0]
                    validation_loss += loss.cpu().item() * x_batch.shape[0]
                    pass
                validation_loss /= batch_count
                pass
            epoch_loss: float = validation_loss
            reporter.progress(
                5.0 + 95.0 * (epoch + 1) / epochs,
                message=f"DNN: Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {validation_loss:.6f}",
                train_loss=train_loss,
                validation_loss=validation_loss,
                epoch_loss=epoch_loss,
                epoch=epoch,
            )
            if epoch_loss < best_loss - self.config_.min_delta:
                best_loss = epoch_loss
                best_state_dict = deepcopy(core.state_dict())
                epochs_without_improvement = 0
                pass
            else:
                epochs_without_improvement += 1
                # * trigger only loss didn't improve
                scheduler.step()
                pass
            if (epochs_without_improvement >= self.config_.patience) and self.config_.early_stop:
                reporter.progress(
                    99.0, 
                    message=f"DNN: Early Stopping Triggered at Epoch {epoch + 1}/{epochs} with Epoch Loss: {epoch_loss:.6f}.",
                    train_loss=train_loss,
                    validation_loss=validation_loss,
                    epoch_loss=epoch_loss,
                    epoch=epoch,
                )
                break
                pass
            pass
        core.load_state_dict(best_state_dict)
        reporter.message(f"DNN: Training Completed. Best Validation Loss: {best_loss:.6f}.", progress=100.0)
        return core
        pass

    pass
