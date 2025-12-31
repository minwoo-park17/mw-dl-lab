"""
ConvLSTM 모델 학습 스크립트

사용법:
    python train.py --data_config config/data_config.yaml --model_config config/model_config.yaml
    python train.py --data_config config/data_config.yaml --model_config config/model_config.yaml --resume checkpoints/last.pth
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.dataset import get_dataloader
from src.model import build_model
from src.utils import (
    load_config,
    merge_configs,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    get_device,
    AverageMeter,
    EarlyStopping
)


def parse_args():
    parser = argparse.ArgumentParser(description='ConvLSTM Training')
    parser.add_argument('--data_config', type=str, default='config/data_config.yaml',
                        help='데이터 config 파일 경로')
    parser.add_argument('--model_config', type=str, default='config/model_config.yaml',
                        help='모델 config 파일 경로')
    parser.add_argument('--resume', type=str, default=None,
                        help='재개할 체크포인트 경로')
    parser.add_argument('--epochs', type=int, default=None,
                        help='학습 에폭 수 (config 덮어쓰기)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='배치 크기 (config 덮어쓰기)')
    parser.add_argument('--lr', type=float, default=None,
                        help='학습률 (config 덮어쓰기)')
    return parser.parse_args()


def train_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    use_amp: bool = True,
    gradient_clip: float = 1.0
):
    """단일 에폭 학습"""
    model.train()

    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Acc')

    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc='Training', leave=False)

    for batch_idx, (sequences, labels) in enumerate(pbar):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision Training
        with autocast(enabled=use_amp):
            outputs = model(sequences)
            loss = criterion(outputs, labels)

        # Backward
        scaler.scale(loss).backward()

        # Gradient Clipping
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        # 메트릭 계산
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())

        # 배치 정확도
        batch_acc = (preds == labels).float().mean().item()

        loss_meter.update(loss.item(), sequences.size(0))
        acc_meter.update(batch_acc, sequences.size(0))

        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'acc': f'{acc_meter.avg:.4f}'
        })

    # 에폭 메트릭
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='binary')

    try:
        epoch_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        epoch_auc = 0.0

    return {
        'loss': loss_meter.avg,
        'accuracy': epoch_acc,
        'f1': epoch_f1,
        'auc': epoch_auc
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device
):
    """검증"""
    model.eval()

    loss_meter = AverageMeter('Loss')

    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc='Validation', leave=False)

    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

        loss_meter.update(loss.item(), sequences.size(0))

    # 메트릭 계산
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='binary')

    try:
        val_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_auc = 0.0

    return {
        'loss': loss_meter.avg,
        'accuracy': val_acc,
        'f1': val_f1,
        'auc': val_auc
    }


def main():
    args = parse_args()

    # Config 로드 및 병합
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)
    config = merge_configs(data_config, model_config)

    # CLI 인자로 config 덮어쓰기
    training_config = config.get('training', {})
    if args.epochs is not None:
        training_config['epochs'] = args.epochs
    if args.batch_size is not None:
        training_config['batch_size'] = args.batch_size
    if args.lr is not None:
        training_config['learning_rate'] = args.lr
    config['training'] = training_config

    # 시드 설정
    seed = config.get('seed', 42)
    set_seed(seed)

    # 디바이스 설정
    device = get_device(config.get('device', 'cuda'))

    # 데이터 로더
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']

    print("\n=== Loading Data ===")
    train_loader = get_dataloader(train_dir, config, is_train=True)
    val_loader = get_dataloader(val_dir, config, is_train=False)

    # 모델
    print("\n=== Building Model ===")
    model = build_model(config)
    model = model.to(device)

    # 손실 함수
    criterion = nn.CrossEntropyLoss()

    # 옵티마이저
    optimizer_config = training_config.get('optimizer', {})
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.0001),
        weight_decay=training_config.get('weight_decay', 0.01),
        betas=tuple(optimizer_config.get('betas', [0.9, 0.999]))
    )

    # 스케줄러
    scheduler_config = training_config.get('scheduler', {})
    epochs = training_config.get('epochs', 50)

    if scheduler_config.get('type') == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=scheduler_config.get('min_lr', 1e-6)
        )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=training_config.get('learning_rate', 0.0001),
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )

    # Mixed Precision
    use_amp = training_config.get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # Early Stopping
    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping = None
    if early_stopping_config.get('enabled', True):
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 10),
            min_delta=early_stopping_config.get('min_delta', 0.001),
            mode='max'  # accuracy 기준
        )

    # 체크포인트 설정
    checkpoint_config = config.get('checkpoint', {})
    save_dir = checkpoint_config.get('save_dir', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)

    # TensorBoard
    log_config = config.get('logging', {})
    log_dir = log_config.get('log_dir', 'logs')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(log_dir, timestamp))

    # 체크포인트에서 재개
    start_epoch = 0
    best_acc = 0.0

    if args.resume:
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch = checkpoint_info.get('epoch', 0) + 1
        best_acc = checkpoint_info.get('accuracy', 0.0)
        print(f"Resuming from epoch {start_epoch}")

    # 학습 루프
    print(f"\n=== Training Start ===")
    print(f"Epochs: {epochs}, Batch size: {training_config.get('batch_size')}")
    print(f"Learning rate: {training_config.get('learning_rate')}")
    print(f"Device: {device}, AMP: {use_amp}")

    gradient_clip = training_config.get('gradient_clip', 1.0)

    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch [{epoch + 1}/{epochs}]")

        # 학습
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp, gradient_clip
        )

        # 검증
        val_metrics = validate(model, val_loader, criterion, device)

        # 스케줄러 업데이트
        if scheduler_config.get('type') == 'cosine':
            scheduler.step()

        # 로깅
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}")
        print(f"  LR: {current_lr:.6f}")

        # TensorBoard 기록
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)

        writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)

        writer.add_scalars('F1', {
            'train': train_metrics['f1'],
            'val': val_metrics['f1']
        }, epoch)

        writer.add_scalars('AUC', {
            'train': train_metrics['auc'],
            'val': val_metrics['auc']
        }, epoch)

        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Best 모델 저장
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            if checkpoint_config.get('save_best', True):
                save_checkpoint(
                    model, optimizer, epoch,
                    val_metrics['loss'], val_metrics['accuracy'],
                    os.path.join(save_dir, 'best.pth'),
                    f1=val_metrics['f1'],
                    auc=val_metrics['auc']
                )
                print(f"  ★ New best model saved (Acc: {best_acc:.4f})")

        # Last 모델 저장
        if checkpoint_config.get('save_last', True):
            save_checkpoint(
                model, optimizer, epoch,
                val_metrics['loss'], val_metrics['accuracy'],
                os.path.join(save_dir, 'last.pth'),
                f1=val_metrics['f1'],
                auc=val_metrics['auc']
            )

        # Early Stopping 체크
        if early_stopping is not None:
            if early_stopping(val_metrics['accuracy']):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

    writer.close()

    print(f"\n=== Training Complete ===")
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"Logs saved to: {os.path.join(log_dir, timestamp)}")


if __name__ == '__main__':
    main()
