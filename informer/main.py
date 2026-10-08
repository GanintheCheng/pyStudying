"""Train and evaluate a compact Informer from scratch on ETTh1."""

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data import InformerDataConfig, InformerWindowDataset, load_etth1
from src.model import InformerForecaster

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def make_decoder_input(y: torch.Tensor, label_len: int) -> torch.Tensor:
    """Keep known decoder history; replace all future true values with zero."""
    return torch.cat([y[:, :label_len, :], torch.zeros_like(y[:, label_len:, :])], dim=1)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, label_len: int, pred_len: int, optimizer: Adam | None = None) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss, total_items = 0.0, 0
    for x, y, x_mark, y_mark in loader:
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        decoder_x = make_decoder_input(y, label_len)
        target = y[:, -pred_len:, :]
        if is_training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_training):
            prediction = model(x, x_mark, decoder_x, y_mark)
            loss = criterion(prediction, target)
            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_items += x.size(0)
    return total_loss / total_items


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, label_len: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions, targets = [], []
    for x, y, x_mark, y_mark in loader:
        x, y, x_mark, y_mark = x.to(device), y.to(device), x_mark.to(device), y_mark.to(device)
        prediction = model(x, x_mark, make_decoder_input(y, label_len), y_mark)
        predictions.append(prediction.cpu().numpy())
        targets.append(y[:, -pred_len:, :].cpu().numpy())
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # A compact setting for first reproduction. Increase epochs/model size later.
    batch_size, epochs, learning_rate = 32, 3, 1e-3
    config = InformerDataConfig(seq_len=96, label_len=48, pred_len=24)
    project_root = Path(__file__).resolve().parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = project_root / "data" / "ETT" / "ETTh1.csv"

    df, values, time_marks, feature_columns, mean, std = load_etth1(data_path, config)
    train_dataset = InformerWindowDataset(values, time_marks, config, config.seq_len, config.train_end)
    val_dataset = InformerWindowDataset(values, time_marks, config, config.train_end, config.val_end)
    test_dataset = InformerWindowDataset(values, time_marks, config, config.val_end, config.test_end)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = InformerForecaster(
        enc_in=len(feature_columns), dec_in=len(feature_columns), c_out=len(feature_columns),
        pred_len=config.pred_len, d_model=64, n_heads=4, e_layers=2, d_layers=1,
        d_ff=128, factor=5, dropout=0.1, distill=True,
    ).to(device)
    optimizer, criterion = Adam(model.parameters(), lr=learning_rate), nn.MSELoss()

    print("===== Informer 配置 =====")
    print("设备：", device)
    print("特征：", feature_columns)
    print("原始测试结束时间：", df["date"].iloc[config.test_end - 1])
    print("窗口数 train / val / test：", len(train_dataset), len(val_dataset), len(test_dataset))
    print("模型参数量：", f"{sum(p.numel() for p in model.parameters()):,}")
    print("Encoder：96 -> 48（两层 Encoder，中间一层 Distilling）")
    print("Decoder：48 个真实历史 + 24 个零占位，直接输出未来 24 步。")

    best_val_loss, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, config.label_len, config.pred_len, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, config.label_len, config.pred_len)
        print(f"epoch={epoch:02d}/{epochs}, train_mse={train_loss:.6f}, val_mse={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss, best_state = val_loss, copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint_path = project_root / "checkpoints" / "best_informer.pt"
    checkpoint_path.parent.mkdir(exist_ok=True)
    torch.save(best_state, checkpoint_path)

    test_mse_scaled = run_epoch(model, test_loader, criterion, device, config.label_len, config.pred_len)
    predicted_scaled, target_scaled = collect_predictions(model, test_loader, device, config.label_len, config.pred_len)
    predicted = predicted_scaled * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    target = target_scaled * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
    mae = np.mean(np.abs(predicted - target))
    mse = np.mean((predicted - target) ** 2)

    print("\n===== 最佳模型的最终测试结果 =====")
    print(f"最佳验证 MSE（标准化尺度）：{best_val_loss:.6f}")
    print(f"测试 MSE（标准化尺度）：    {test_mse_scaled:.6f}")
    print(f"测试 MAE（原始尺度）：      {mae:.4f}")
    print(f"测试 MSE（原始尺度）：      {mse:.4f}")
    print(f"测试 RMSE（原始尺度）：     {np.sqrt(mse):.4f}")
    print("模型检查点：", checkpoint_path)

    print("\n===== 一个测试窗口：未来第 1 小时预测 =====")
    for name, actual, forecast in zip(feature_columns, target[0, 0], predicted[0, 0]):
        print(f"{name:>4}: actual={actual:8.3f}, prediction={forecast:8.3f}")


if __name__ == "__main__":
    main()
