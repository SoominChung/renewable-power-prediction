from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple

import os
import yaml
import json
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import wandb
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 모델 import (위에서 정의한 모델)
#from complete_solar_prediction_model import create_solar_prediction_model, SolarPowerPredictor

def create_model_by_name(model_name: str, cfg: Dict, device: str):
    """모델명에 따라 모델 생성"""
    if model_name.lower() == "transformer":
        from complete_solar_prediction_model import create_solar_prediction_model
        return create_solar_prediction_model(
            historical_features=cfg["model"]["historical_features"],
            weather_features=cfg["model"]["weather_features"],
            plant_features=cfg["model"]["plant_features"],
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            device=device
        )
    elif model_name.lower() == "simple":
        from simple_solar_model import create_simple_solar_model
        return create_simple_solar_model(
            historical_features=cfg["model"]["historical_features"],
            weather_features=cfg["model"]["weather_features"],
            plant_features=cfg["model"]["plant_features"],
            hidden_dim=cfg["model"]["d_model"],  # d_model을 hidden_dim으로 사용
            device=device
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}. Use 'transformer' or 'simple'")

# ─── 0) 설정 로드 및 W&B 인증 ───────────────────────────
def load_config(config_path: str = "config.yaml") -> Dict:
    """설정 파일 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_wandb(cfg: Dict):
    """W&B 설정 및 로그인"""
    os.environ["WANDB_API_KEY"] = cfg["wandb"]["api_key"]
    os.environ["WANDB_ENTITY"] = cfg["wandb"]["entity"]
    os.environ["WANDB_PROJECT"] = cfg["wandb"]["project"]
    wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)


def set_seed(seed: int):
    """시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── 1) 데이터셋 클래스 ─────────────────────────────────
class SolarDataset(Dataset):
    """태양광 발전량 예측을 위한 데이터셋"""
    
    def __init__(
        self, 
        data_dir: str, 
        scenario: str = "14시", 
        split: str = "train",
        device: str = "cuda:4"
    ):
        self.data_dir = data_dir
        self.scenario = scenario
        self.split = split
        self.device = device
        
        # 데이터 로드
        self._load_data()
        self._load_metadata()
        
        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.X_historical)}")
        print(f"  Historical shape: {self.X_historical.shape}")
        print(f"  Forecast shape: {self.X_forecast.shape}")
        print(f"  Meta shape: {self.X_meta.shape}")
        print(f"  Target shape: {self.y.shape}")
        
    def _load_data(self):
        """데이터 파일 로드"""
        split_dir = os.path.join(self.data_dir, self.scenario, self.split)

        # 데이터 로드 및 float32, torch.tensor로 변환하여 메모리에 유지
        self.X_historical = torch.tensor(np.load(os.path.join(split_dir, 'X_historical.npy'), allow_pickle=True), dtype=torch.float32)
        self.X_forecast = torch.tensor(np.load(os.path.join(split_dir, 'X_forecast.npy'), allow_pickle=True), dtype=torch.float32)
        self.X_meta = torch.tensor(np.load(os.path.join(split_dir, 'X_meta.npy'), allow_pickle=True), dtype=torch.float32)
        self.y = torch.tensor(np.load(os.path.join(split_dir, 'y.npy'), allow_pickle=True), dtype=torch.float32)
        self.y_original = torch.tensor(np.load(os.path.join(split_dir, 'y_original.npy'), allow_pickle=True), dtype=torch.float32)
        self.plants = np.load(os.path.join(split_dir, 'plants.npy'), allow_pickle=True) # plants는 문자열이므로 numpy 유지
        self.padding_mask = torch.tensor(np.load(os.path.join(split_dir, 'padding_mask.npy'), allow_pickle=True), dtype=torch.bool)

        # 여기서 self.X_historical.to(self.device) 등을 하면 GPU 메모리도 초기화 시점에 사용
        # 하지만 DataLoader가 알아서 GPU로 옮겨주므로 __getitem__에서는 .to(device) 제거
        print(f"  Data loaded into tensors on CPU.")

        
    def _load_metadata(self):
        """발전소 메타데이터 로드"""
        try:
            self.meta_df = pd.read_csv('/home/local/soominchung_991007/EF/code/tmp/data/solar_energy/meta_data.csv')
            print(f"Metadata loaded: {len(self.meta_df)} plants")
        except:
            print("Warning: meta_data.csv not found. Capacity info will be unavailable.")
            self.meta_df = None
    
    def get_plant_capacity(self, plant_name: str) -> float:
        """발전소 용량 조회 (kW 단위)"""
        if self.meta_df is None:
            return 1000.0  # 기본값
        
        try:
            capacity_mw = self.meta_df[self.meta_df['name'] == plant_name]['용량(MW)'].iloc[0]
            return float(capacity_mw) * 1000  # MW to kW
        except:
            return 1000.0  # 기본값
    
    def denormalize_predictions(self, normalized_y: np.ndarray, plant_names: np.ndarray) -> np.ndarray:
        """정규화된 예측값을 원본 스케일로 복원"""
        denormalized = np.zeros_like(normalized_y)
        
        for i, plant_name in enumerate(plant_names):
            capacity_kw = self.get_plant_capacity(plant_name)
            # 정규화가 용량 기반이라면: 이용률 * 용량 = 발전량
            denormalized[i] = normalized_y[i] * capacity_kw
            
        return denormalized
    
    def __len__(self):
        return len(self.X_historical)
    
    def __getitem__(self, idx):
        return {
            'historical': self.X_historical[idx], # .to(self.device) 제거
            'forecast': self.X_forecast[idx],   # .to(self.device) 제거
            'meta': self.X_meta[idx],         # .to(self.device) 제거
            'target': self.y[idx],
            'target_original': self.y_original[idx],
            'plant_name': self.plants[idx],
            'padding_mask': self.padding_mask[idx]
        }        
        '''
        return {
            'historical': torch.tensor(self.X_historical[idx], dtype=torch.float32),
            'forecast': torch.tensor(self.X_forecast[idx], dtype=torch.float32),
            'meta': torch.tensor(self.X_meta[idx], dtype=torch.float32),
            'target': torch.tensor(self.y[idx], dtype=torch.float32),  # 정규화된 값
            'target_original': torch.tensor(self.y_original[idx], dtype=torch.float32),  # 원본 값
            'plant_name': self.plants[idx],
            'padding_mask': torch.tensor(self.padding_mask[idx], dtype=torch.bool)
        }
        '''


# ─── 2) 데이터로더 생성 ─────────────────────────────────
def create_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """데이터로더 생성"""
    data_dir = cfg["data"]["root_dir"]
    batch_size = cfg["training"]["batch_size"]
    device = cfg["training"]["device"]
    
    # 14시와 20시 시나리오 데이터셋을 통합
    datasets = {}
    
    for scenario in ["14시", "20시"]:
        for split in ["train", "valid", "test", "external_test"]:
            key = f"{scenario}_{split}"
            datasets[key] = SolarDataset(
                data_dir=data_dir,
                scenario=scenario,
                split=split,
                device=device
            )
    
    # 14시와 20시 데이터를 각 split별로 결합
    def combine_datasets(dataset1, dataset2):
        """두 데이터셋을 결합"""
        combined_data = {}
        
        # 모든 데이터 결합
        for key in ['X_historical', 'X_forecast', 'X_meta', 'y', 'y_original', 'plants', 'padding_mask']:
            data1 = getattr(dataset1, key)
            data2 = getattr(dataset2, key)
            combined_data[key] = np.concatenate([data1, data2], axis=0)
        
        # 새로운 데이터셋 객체 생성
        combined_dataset = SolarDataset.__new__(SolarDataset)
        combined_dataset.data_dir = dataset1.data_dir
        combined_dataset.scenario = "combined"
        combined_dataset.split = dataset1.split
        combined_dataset.device = dataset1.device
        combined_dataset.meta_df = dataset1.meta_df
        
        # 결합된 데이터 할당
        for key, value in combined_data.items():
            setattr(combined_dataset, key, value)
        
        return combined_dataset
    
    # 데이터셋 결합
    train_dataset = combine_datasets(datasets["14시_train"], datasets["20시_train"])
    valid_dataset = combine_datasets(datasets["14시_valid"], datasets["20시_valid"])
    test_dataset = combine_datasets(datasets["14시_test"], datasets["20시_test"])
    external_test_dataset = combine_datasets(datasets["14시_external_test"], datasets["20시_external_test"])
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    external_test_loader = DataLoader(
        external_test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, valid_loader, test_loader, external_test_loader


# ─── 3) 손실 함수 정의 ─────────────────────────────────
class SolarPredictionLoss(nn.Module):
    """태양광 발전량 예측을 위한 커스텀 손실 함수"""
    
    def __init__(self, loss_type: str = "MSELoss", use_capacity_weighting: bool = True, 
                 use_denormalized_loss: bool = True, per_weight: float = 0.5):
        super().__init__()
        self.loss_type = loss_type
        self.use_capacity_weighting = use_capacity_weighting
        self.use_denormalized_loss = use_denormalized_loss
        self.per_weight = per_weight  
        
        # 기본 손실 함수
        if loss_type == "MSELoss":
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == "L1Loss":
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == "SmoothL1Loss":
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        elif loss_type == "PredictionErrorRate":
            self.base_loss = None  # 커스텀 손실 함수 사용
        elif loss_type == "MSE_PER_Weighted":  # 👈 추가
            self.base_loss = nn.MSELoss(reduction='none')
        else:
            self.base_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                plant_capacities: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch, 24] - 정규화된 예측값 (이용률, 0~1)
            targets: [batch, 24] - 정규화된 실제값 (이용률, 0~1)
            plant_capacities: [batch] - 발전소 용량 (kW)
        """
        if self.loss_type == "PredictionErrorRate":
            return self._prediction_error_rate_loss(predictions, targets, plant_capacities)
        elif self.loss_type == "MSE_PER_Weighted":  # 👈 추가
            return self._mse_per_weighted_loss(predictions, targets, plant_capacities)
        
        if self.use_denormalized_loss:
            # 정규화된 값을 실제 발전량(kW)으로 역정규화
            capacity_expanded = plant_capacities.unsqueeze(1)  # [batch, 1]
            pred_power = predictions * capacity_expanded  # [batch, 24] - 실제 발전량 (kW)
            target_power = targets * capacity_expanded    # [batch, 24] - 실제 발전량 (kW)
            
            # 실제 발전량으로 손실 계산
            loss = self.base_loss(pred_power, target_power)  # [batch, 24]

            # 용량으로 정규화하여 스케일 조정
            #loss = loss / (capacity_expanded ** 2)  # 제곱으로 나누어 스케일 감소            
            loss = loss / (capacity_expanded) 
            
            # 용량 기반 가중치 (선택사항)
            if self.use_capacity_weighting:
                # 큰 발전소일수록 높은 가중치, 하지만 이미 역정규화했으므로 약하게 적용
                capacity_weights = torch.sqrt(plant_capacities / plant_capacities.mean())
                capacity_weights = capacity_weights.unsqueeze(1)  # [batch, 1]
                loss = loss * capacity_weights
            
        else:
            # 기존 방식: 정규화된 값으로 손실 계산
            loss = self.base_loss(predictions, targets)  # [batch, 24]
            
            if self.use_capacity_weighting:
                # 용량 기반 가중치 (큰 발전소일수록 높은 가중치)
                capacity_weights = plant_capacities / plant_capacities.mean()
                capacity_weights = capacity_weights.unsqueeze(1)  # [batch, 1]
                loss = loss * capacity_weights
        
        # 시간축 평균 후 배치 평균
        return loss.mean()
    
    def _prediction_error_rate_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                   plant_capacities: torch.Tensor, threshold_ratio: float = 0.1) -> torch.Tensor:
        """재생에너지 발전량 예측제도 오차율 기반 손실 함수"""
        # 이용률을 실제 발전량으로 변환
        capacity_expanded = plant_capacities.unsqueeze(1)  # [batch, 1]
        pred_power = predictions * capacity_expanded  # [batch, 24]
        target_power = targets * capacity_expanded    # [batch, 24]
        
        # 임계값 이상인 시간대만 마스킹
        threshold_power = plant_capacities * threshold_ratio
        threshold_expanded = threshold_power.unsqueeze(1)  # [batch, 1]
        mask = (target_power >= threshold_expanded).float()  # [batch, 24]
        
        # 절대 오차 계산
        abs_error = torch.abs(target_power - pred_power)  # [batch, 24]
        
        # 마스킹된 오차 합산
        masked_error_sum = (abs_error * mask).sum(dim=1)  # [batch]
        
        # 오차율 계산 (%) - 0으로 나누기 방지
        error_rate = (masked_error_sum / (plant_capacities + 1e-8)) * 100.0  # [batch]
        
        return error_rate.mean()

    def _mse_per_weighted_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                            plant_capacities: torch.Tensor) -> torch.Tensor:
        """MSE + PER 가중합 손실"""
        # MSE 손실 계산
        if self.use_denormalized_loss:
            capacity_expanded = plant_capacities.unsqueeze(1)
            pred_power = predictions * capacity_expanded
            target_power = targets * capacity_expanded
            mse_loss = self.base_loss(pred_power, target_power)
            mse_loss = mse_loss / capacity_expanded  # 스케일 조정
            mse_loss = mse_loss.mean()
        else:
            mse_loss = self.base_loss(predictions, targets).mean()
        
        # PER 손실 계산
        per_loss = self._prediction_error_rate_loss(predictions, targets, plant_capacities)
        
        # 가중합 (PER을 100으로 나누어 스케일 맞춤)
        weighted_loss = (1 - self.per_weight) * mse_loss + self.per_weight * (per_loss / 100.0)
        
        return weighted_loss

def print_sample_predictions(model, train_loader, valid_loader, device, epoch, num_samples=3):
    """샘플 예측값과 실제값 출력"""
    model.eval()
    
    print(f"\n📊 Sample Predictions - Epoch {epoch}")
    print("=" * 60)
    
    # Train 샘플
    with torch.no_grad():
        train_batch = next(iter(train_loader))
        
        historical = train_batch['historical'].to(device)[:num_samples]
        forecast = train_batch['forecast'].to(device)[:num_samples]
        meta = train_batch['meta'].to(device)[:num_samples]
        target = train_batch['target'].to(device)[:num_samples]
        padding_mask = train_batch['padding_mask'].to(device)[:num_samples]
        plant_names = train_batch['plant_name'][:num_samples]
        
        # 용량 정보
        capacities = []
        for plant_name in plant_names:
            capacity = train_loader.dataset.get_plant_capacity(plant_name)
            capacities.append(capacity)
        capacities = torch.tensor(capacities, device=device, dtype=torch.float32)
        
        # 예측
        outputs = model(historical, forecast, meta, padding_mask)
        predictions = outputs['predictions']
        
        if epoch % 5 == 0 or epoch == 1:
            print("🚂 TRAIN Samples:")
            for i in range(num_samples):
                pred_power = predictions[i] * capacities[i]
                target_power = target[i] * capacities[i]
                
                # 몇 개 시간대만 출력 (0, 6, 12, 18시)
                sample_hours = [0, 6, 12, 18]
                
                print(f"  Plant: {plant_names[i][:15]:15s} | Capacity: {capacities[i].item():6.0f}kW")
                for hour in sample_hours:
                    pred_val = pred_power[hour].item()
                    target_val = target_power[hour].item()
                    error = abs(pred_val - target_val)
                    print(f"    Hour {hour:2d}: Pred={pred_val:6.1f}kW, True={target_val:6.1f}kW, Error={error:6.1f}kW")
        
    # Valid 샘플
    with torch.no_grad():
        valid_batch = next(iter(valid_loader))
        
        historical = valid_batch['historical'].to(device)[:num_samples]
        forecast = valid_batch['forecast'].to(device)[:num_samples]
        meta = valid_batch['meta'].to(device)[:num_samples]
        target = valid_batch['target'].to(device)[:num_samples]
        padding_mask = valid_batch['padding_mask'].to(device)[:num_samples]
        plant_names = valid_batch['plant_name'][:num_samples]
        
        # 용량 정보
        capacities = []
        for plant_name in plant_names:
            capacity = valid_loader.dataset.get_plant_capacity(plant_name)
            capacities.append(capacity)
        capacities = torch.tensor(capacities, device=device, dtype=torch.float32)
        
        # 예측
        outputs = model(historical, forecast, meta, padding_mask)
        predictions = outputs['predictions']
        
        if epoch % 5 == 0 or epoch == 1:
            print("\n✅ VALID Samples:")
            for i in range(num_samples):
                pred_power = predictions[i] * capacities[i]
                target_power = target[i] * capacities[i]
                
                sample_hours = [0, 6, 12, 18]
                
                print(f"  Plant: {plant_names[i][:15]:15s} | Capacity: {capacities[i].item():6.0f}kW")
                for hour in sample_hours:
                    pred_val = pred_power[hour].item()
                    target_val = target_power[hour].item()
                    error = abs(pred_val - target_val)
                    print(f"    Hour {hour:2d}: Pred={pred_val:6.1f}kW, True={target_val:6.1f}kW, Error={error:6.1f}kW")
        
    print("=" * 60)

# ─── 4) 평가 메트릭 ─────────────────────────────────
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, 
                     plant_capacities: np.ndarray) -> Dict[str, float]:
    """평가 메트릭 계산"""
    # 이용률을 실제 발전량으로 변환
    pred_power = predictions * plant_capacities[:, np.newaxis]
    target_power = targets * plant_capacities[:, np.newaxis]
    
    # 기본 메트릭
    mse = np.mean((pred_power - target_power) ** 2)
    mae = np.mean(np.abs(pred_power - target_power))
    rmse = np.sqrt(mse)
    
    # MAPE (0이 아닌 값들에 대해서만)
    non_zero_mask = target_power != 0
    mape = np.mean(np.abs((target_power[non_zero_mask] - pred_power[non_zero_mask]) / target_power[non_zero_mask])) * 100
    
    # 예측 정확도 (10% 오차 내)
    relative_error = np.abs((target_power - pred_power) / (target_power + 1e-8))
    accuracy_10 = np.mean(relative_error <= 0.1) * 100
    
    # 발전량 예측제도 오차율 (10% 이상 구간)
    threshold_power = plant_capacities * 0.1
    mask = target_power >= threshold_power[:, np.newaxis]
    
    if mask.any():
        masked_error = np.abs(target_power - pred_power) * mask
        error_sum = masked_error.sum(axis=1)
        error_rate = (error_sum / plant_capacities) * 100
        prediction_error_rate = error_rate.mean()
    else:
        prediction_error_rate = 0.0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Accuracy@10%': accuracy_10,
        'PredictionErrorRate': prediction_error_rate
    }

def save_predictions(model, data_loader: DataLoader, 
                    device: str, result_dir: str, split_name: str) -> Dict[str, float]:
    """예측값 저장 및 메트릭 계산"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_targets_original = []
    all_capacities = []
    all_plant_names = []
    
    # 예측 수행
    with torch.no_grad():
        for batch in data_loader:
            historical = batch['historical'].to(device)
            forecast = batch['forecast'].to(device)
            meta = batch['meta'].to(device)
            target = batch['target'].to(device)
            target_original = batch['target_original'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            plant_names = batch['plant_name']
            
            # 발전소 용량 정보
            capacities = []
            for plant_name in plant_names:
                capacity = data_loader.dataset.get_plant_capacity(plant_name)
                capacities.append(capacity)
            capacities = torch.tensor(capacities, device=device, dtype=torch.float32)
            
            # 예측
            outputs = model(historical, forecast, meta, padding_mask)
            predictions = outputs['predictions']
            
            # 데이터 수집
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_targets_original.append(target_original.cpu().numpy())
            all_capacities.append(capacities.cpu().numpy())
            all_plant_names.extend(plant_names)
    
    # 배열 결합
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    targets_original = np.concatenate(all_targets_original, axis=0)
    capacities = np.concatenate(all_capacities, axis=0)
    
    # 역정규화된 예측값 계산
    predictions_denormalized = predictions * capacities[:, np.newaxis]
    
    # 예측값 저장
    np.save(os.path.join(result_dir, f'{split_name}_predictions_normalized.npy'), predictions)
    np.save(os.path.join(result_dir, f'{split_name}_predictions_denormalized.npy'), predictions_denormalized)
    np.save(os.path.join(result_dir, f'{split_name}_targets_normalized.npy'), targets)
    np.save(os.path.join(result_dir, f'{split_name}_targets_original.npy'), targets_original)
    np.save(os.path.join(result_dir, f'{split_name}_capacities.npy'), capacities)
    np.save(os.path.join(result_dir, f'{split_name}_plant_names.npy'), np.array(all_plant_names))
    
    # 메트릭 계산
    metrics = calculate_metrics(predictions, targets, capacities)
    
    print(f"✅ {split_name} predictions saved to {result_dir}")
    print(f"   - Shape: {predictions.shape}")
    print(f"   - Metrics: {metrics}")
    
    return metrics

# ─── 5) 훈련 함수 ─────────────────────────────────
def train_epoch(model, train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, criterion: SolarPredictionLoss, device: str) -> float:
    """한 에포크 훈련"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        # 데이터 이동
        historical = batch['historical'].to(device)
        forecast = batch['forecast'].to(device)
        meta = batch['meta'].to(device)
        target = batch['target'].to(device)  # 정규화된 값 사용
        padding_mask = batch['padding_mask'].to(device)
        plant_names = batch['plant_name']
        
        # 발전소 용량 정보 가져오기
        capacities = []
        for plant_name in plant_names:
            capacity = train_loader.dataset.get_plant_capacity(plant_name)
            capacities.append(capacity)
        capacities = torch.tensor(capacities, device=device, dtype=torch.float32)
        
        # Forward pass
        optimizer.zero_grad()
        
        outputs = model(historical, forecast, meta, padding_mask)
        predictions = outputs['predictions']
        
        # 손실 계산 - 이제 모든 경우에 capacity를 전달
        loss = criterion(predictions, target, capacities)
        
        # NaN 체크
        if torch.isnan(loss):
            print(f"NaN loss detected! Skipping batch.")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return 0.0
    
    return total_loss / num_batches


def validate_epoch(model, valid_loader: DataLoader, 
                  criterion: SolarPredictionLoss, device: str) -> Tuple[float, Dict[str, float]]:
    """검증"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_capacities = []
    
    with torch.no_grad():
        for batch in valid_loader:
            # 데이터 이동
            historical = batch['historical'].to(device)
            forecast = batch['forecast'].to(device)
            meta = batch['meta'].to(device)
            target = batch['target'].to(device)  # 정규화된 값 사용
            padding_mask = batch['padding_mask'].to(device)
            plant_names = batch['plant_name']
            
            # 발전소 용량 정보
            capacities = []
            for plant_name in plant_names:
                capacity = valid_loader.dataset.get_plant_capacity(plant_name)
                capacities.append(capacity)
            capacities = torch.tensor(capacities, device=device, dtype=torch.float32)
            
            # Forward pass
            outputs = model(historical, forecast, meta, padding_mask)
            predictions = outputs['predictions']
            
            # 손실 계산 - 일관성 있게 capacity 전달
            loss = criterion(predictions, target, capacities)
            
            # NaN 체크
            if not torch.isnan(loss):
                total_loss += loss.item()
                
                # 메트릭 계산을 위해 데이터 수집
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                all_capacities.append(capacities.cpu().numpy())
    
    if len(all_predictions) == 0:
        return float('inf'), {}
    
    # 메트릭 계산
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    capacities = np.concatenate(all_capacities, axis=0)
    
    metrics = calculate_metrics(predictions, targets, capacities)
    
    return total_loss / len(valid_loader), metrics


# ─── 6) 메인 훈련 함수 ─────────────────────────────────
def train_solar_prediction_model(cfg: Dict):
    """태양광 발전량 예측 모델 훈련"""

    # 결과 저장 디렉토리 생성
    result_dir = create_result_directory(cfg)    
    
    # 시드 설정
    set_seed(cfg["training"]["seed"])
    
    # 디바이스 설정
    device = cfg["training"]["device"]
    print(f"Using device: {device}")
    
    # 데이터로더 생성
    print("Creating dataloaders...")
    train_loader, valid_loader, test_loader, external_test_loader = create_dataloaders(cfg)
    
    # 모델 생성
    model_name = cfg["model"]["name"]
    print(f"Creating {model_name} model...")
    model = create_model_by_name(model_name, cfg, device)


    # 모델 가중치 초기화 개선
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.1)  # 더 작은 gain
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    model.apply(init_weights)
    print("Applied improved weight initialization")        

    # 옵티마이저 생성
    optimizer_class = getattr(torch.optim, cfg["training"]["optimizer"])
    optimizer = optimizer_class(
        model.parameters(), 
        lr=cfg["training"]["learning_rate"],
        **cfg["training"]["optimizer_params"]
    )
    
    # 스케줄러 생성
    scheduler_class = getattr(torch.optim.lr_scheduler, cfg["training"]["scheduler"])
    scheduler = scheduler_class(optimizer, **cfg["training"]["scheduler_params"])
    
    # 손실 함수 생성 - 통합된 방식으로 수정
    criterion = SolarPredictionLoss(
        loss_type=cfg["training"]["loss"],
        use_capacity_weighting=cfg["training"].get("use_capacity_weighting", True),
        use_denormalized_loss=cfg["training"].get("use_denormalized_loss", True)
    )
    
    # W&B 초기화
    model_name = cfg["model"]["name"]
    run_name = (f"Solar{model_name.capitalize()}_"
            f"{cfg['model']['d_model']}d_"
            f"{cfg['training']['batch_size']}batch_"
            f"{cfg['training']['learning_rate']:.0e}_"
            f"{cfg['training']['loss']}_"
            f"decay{cfg['training']['optimizer_params']['weight_decay']}")

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        name=run_name,
        config=cfg
    )
    
    # 모델 구조 로깅
    wandb.watch(model, log="all", log_freq=100)
    
    # 첫 번째 배치로 모델 출력 확인
    print("Checking model output...")
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        historical = sample_batch['historical'].to(device)[:2]  # 작은 배치로 테스트
        forecast = sample_batch['forecast'].to(device)[:2]
        meta = sample_batch['meta'].to(device)[:2]
        target = sample_batch['target'].to(device)[:2]
        padding_mask = sample_batch['padding_mask'].to(device)[:2]
        
        outputs = model(historical, forecast, meta, padding_mask)
        predictions = outputs['predictions']
        
        print(f"Sample predictions shape: {predictions.shape}")
        print(f"Sample targets shape: {target.shape}")
        print(f"Sample predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"Sample targets range: [{target.min():.4f}, {target.max():.4f}]")

        # 역정규화된 값도 확인
        capacities = []
        for plant_name in sample_batch['plant_name'][:2]:
            capacity = train_loader.dataset.get_plant_capacity(plant_name)
            capacities.append(capacity)
        capacities = torch.tensor(capacities, device=device, dtype=torch.float32)
        
        pred_power = predictions * capacities.unsqueeze(1)
        target_power = target * capacities.unsqueeze(1)
        print(f"Sample pred_power range: [{pred_power.min():.2f}, {pred_power.max():.2f}] kW")
        print(f"Sample target_power range: [{target_power.min():.2f}, {target_power.max():.2f}] kW")
        print(f"Sample capacities: {capacities.cpu().numpy()}")
        
        # 손실 함수 테스트
        test_loss = criterion(predictions, target, capacities)
        print(f"Sample loss: {test_loss.item():.4f}")        
    
    # 훈련 루프
    best_val_loss = float('inf')
    patience_counter = 0
    best_metrics = None
    
    print("Starting training...")
    
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        # 훈련
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 검증
        val_loss, val_metrics = validate_epoch(model, valid_loader, criterion, device)
        
        print_sample_predictions(model, train_loader, valid_loader, device, epoch, num_samples=3)

        # 스케줄러 스텝
        if cfg["training"]["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 로깅
        current_lr = optimizer.param_groups[0]['lr']
        log_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr
        }
        
        if val_metrics:
            log_data.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        wandb.log(log_data)
        
        # 출력
        if epoch % 5 == 0 or epoch == 1:
            if val_metrics:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val RMSE: {val_metrics.get('RMSE', 0):.2f} | "
                      f"Val ErrorRate: {val_metrics.get('PredictionErrorRate', 0):.2f}% | "
                      f"LR: {current_lr:.2e}")
            else:
                print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
        
        # Early stopping 및 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics.copy() if val_metrics else {}
            patience_counter = 0
            
            # 모델 저장
            model_path = os.path.join(result_dir, f"best_solar_model_{run_name}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': cfg
            }, model_path)
            
            print(f"💾 Best model saved at epoch {epoch} with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
            if (cfg["training"]["early_stopping"]["use"] and 
                patience_counter >= cfg["training"]["early_stopping"]["patience"]):
                print(f"⏰ Early stopping at epoch {epoch}")
                break
    
    # 최종 테스트
    print("\n🧪 Evaluating on test set...")
    model_path = os.path.join(result_dir, f"best_solar_model_{run_name}.pth")
    model.load_state_dict(torch.load(model_path,weights_only=False)['model_state_dict'])

    # 테스트 예측값 저장 및 메트릭 계산
    valid_metrics = save_predictions(model, valid_loader, device, result_dir, "valid")
    test_metrics = save_predictions(model, test_loader, device, result_dir, "test")
    external_test_metrics = save_predictions(model, external_test_loader, device, result_dir, "external_test")

    # 손실값 계산
    test_loss, test_metrics = validate_epoch(model, test_loader, criterion, device)
    external_test_loss, external_test_metrics = validate_epoch(model, external_test_loader, criterion, device)

    # 설정 파일도 저장
    config_save_path = os.path.join(result_dir, "config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ Config saved to {config_save_path}")    
    
    # 최종 결과 로깅
    final_results = {
        "final_test_loss": test_loss,
        "final_external_test_loss": external_test_loss,
    }
    
    if test_metrics:
        final_results.update({f"test_{k}": v for k, v in test_metrics.items()})
    if external_test_metrics:
        final_results.update({f"external_test_{k}": v for k, v in external_test_metrics.items()})
    
    wandb.log(final_results)
    
    print(f"\n🎉 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test metrics: {test_metrics}")
    print(f"External test metrics: {external_test_metrics}")
    
    wandb.finish()
    
    return model, best_metrics, test_metrics, external_test_metrics


# ─── 7) 설정 파일 생성 ─────────────────────────────────
def create_result_directory(cfg: Dict) -> str:
    """결과 저장 디렉토리 생성"""
    current_time_str = datetime.now().strftime("%m%d_%H%M%S") 

    # 디렉토리 이름 생성
    dir_name = (f"{cfg['model']['name']}_"
                f"{cfg['model']['d_model']}d_model_"
                f"{cfg['model']['num_encoder_layers']}n_layers_"
                f"{cfg['training']['learning_rate']}lr_"
                f"{cfg['training']['batch_size']}batch_"
                f"{cfg['training']['loss']}_"
                f"{current_time_str}"
                )
    
    result_dir = os.path.join("./result", dir_name)
    os.makedirs(result_dir, exist_ok=True)
    
    print(f"Results will be saved to: {result_dir}")
    return result_dir

def create_config_file():
    """설정 파일 생성"""
    config = {
        "data": {
            "root_dir": "/home/local/soominchung_991007/EF/code/tmp/data/normalized_forecast_modeling_data",
            "meta_path": "/home/local/soominchung_991007/EF/code/tmp/data/solar_energy/meta_data.csv"
        },
        "model": {
            "name": "simple", # "name": "simple",  # 👈 여기 추가: "transformer" 또는 "simple"
            "historical_features": 54,
            "weather_features": 5,
            "plant_features": 10,
            "d_model": 64, #256
            "nhead": 4,
            "num_encoder_layers": 1
        },
        "training": {
            "seed": 42,
            "device": "cuda:6",
            "learning_rate": 5e-5,
            "batch_size": 64,
            "epochs": 100,
            "optimizer": "AdamW",
            "optimizer_params": {
                "weight_decay": 0.1,
                "eps": 1e-8
            },
            "scheduler": "ReduceLROnPlateau",
            "scheduler_params": {
                "mode": "min",
                "factor": 0.3,
                "patience": 10,
                "min_lr": 1e-6
            },
            "loss": "MSELoss",  # or "L1Loss", "SmoothL1Loss", "PredictionErrorRate"
            "use_capacity_weighting": False,
            "use_denormalized_loss": True,  # 역정규화 손실 사용
            "early_stopping": {
                "use": True,
                "patience": 10
            }
        },
        "wandb": {
            "api_key": "a8af4997e63da6343549fc1212570a2d1c274303",
            "entity": "soomin200-seoul-national-university",
            "project": "solar_prediction_multimodal"
        }
    }
    
    with open("solar_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print("Created solar_config.yaml")


# ─── 8) 메인 실행 ─────────────────────────────────────

'''
## 하나만 돌릴 때
# 설정 파일 생성 (처음 실행 시)
#if not os.path.exists("solar_config.yaml"):
create_config_file()

# 설정 로드
cfg = load_config("solar_config.yaml")

# W&B 설정
setup_wandb(cfg)

# 훈련 실행
model, best_metrics, test_metrics, external_test_metrics = train_solar_prediction_model(cfg)

print("\n🌞 Solar Power Prediction Training Completed! 🌞")
'''

# ===== 여러 실험 자동 실행 =====

# 기본 설정 생성
create_config_file()
base_config = load_config("solar_config.yaml")
'''
experiments = [
    # ===== Simple 모델 (기본 성능 확인) =====
    # (실험명, model_name, learning_rate, batch_size, d_model, loss, weight_decay)
    ("simple_baseline", "simple", 1e-4, 32, 64, "MSELoss", 0.1),
    ("simple_policy_per", "simple", 1e-4, 32, 64, "PredictionErrorRate", 0.1),
    ("simple_combined", "simple", 1e-4, 32, 64, "MSE_PER_Weighted", 0.1),
    
    # ===== Transformer vs Simple 직접 비교 (동일 조건) =====
    ("transformer_baseline", "transformer", 5e-5, 32, 64, "MSELoss", 0.1),
    ("transformer_policy_per", "transformer", 5e-5, 32, 64, "PredictionErrorRate", 0.1),
    ("transformer_combined", "transformer", 5e-5, 32, 64, "MSE_PER_Weighted", 0.1),
    
    # ===== 학습률 실험 (각 모델별 최적화) =====
    ("simple_higher_lr", "simple", 2e-4, 32, 64, "MSE_PER_Weighted", 0.1),
    ("transformer_higher_lr", "transformer", 1e-4, 32, 64, "MSE_PER_Weighted", 0.1),
    
    # ===== 작은 모델 크기 실험 =====
    ("simple_tiny", "simple", 1e-4, 32, 32, "MSE_PER_Weighted", 0.05),
    ("transformer_tiny", "transformer", 5e-5, 32, 32, "MSE_PER_Weighted", 0.05),
    
    # ===== 중간 크기 모델 (좋은 결과 나오면) =====
    ("simple_medium", "simple", 1e-4, 32, 128, "MSE_PER_Weighted", 0.05),
    ("transformer_medium", "transformer", 5e-5, 32, 128, "MSE_PER_Weighted", 0.05),
    
    # ===== 배치 크기 실험 =====
    ("simple_large_batch", "simple", 5e-5, 64, 64, "MSE_PER_Weighted", 0.1),
    ("transformer_large_batch", "transformer", 3e-5, 64, 64, "MSE_PER_Weighted", 0.1),
    
    # ===== 최종 도전 (큰 모델) =====
    ("simple_large", "simple", 8e-5, 16, 256, "MSE_PER_Weighted", 0.03),
    ("transformer_large", "transformer", 3e-5, 16, 256, "MSE_PER_Weighted", 0.03),
]
'''

experiments = [
    # === nhead 실험 (layers=2 고정) ===
    ("transformer_256d_2head_2layer_mse", "transformer", 3e-5, 16, 256, "MSELoss", 0.03, 2, 2),
    ("transformer_256d_2head_2layer_weighted", "transformer", 3e-5, 16, 256, "MSE_PER_Weighted", 0.03, 2, 2),
    
    ("transformer_256d_4head_2layer_mse", "transformer", 3e-5, 16, 256, "MSELoss", 0.03, 4, 2),
    ("transformer_256d_4head_2layer_weighted", "transformer", 3e-5, 16, 256, "MSE_PER_Weighted", 0.03, 4, 2),
    
    ("transformer_256d_6head_2layer_mse", "transformer", 3e-5, 16, 256, "MSELoss", 0.03, 6, 2),
    
    #####
    ("transformer_256d_6head_2layer_weighted", "transformer", 3e-5, 16, 256, "MSE_PER_Weighted", 0.03, 6, 2),
    
    # === num_encoder_layers 실험 (nhead=4 고정) ===
    ("transformer_256d_4head_1layer_mse", "transformer", 3e-5, 16, 256, "MSELoss", 0.03, 4, 1),
    ("transformer_256d_4head_1layer_weighted", "transformer", 3e-5, 16, 256, "MSE_PER_Weighted", 0.03, 4, 1),
    
    ("transformer_256d_4head_3layer_mse", "transformer", 3e-5, 16, 256, "MSELoss", 0.03, 4, 3),
    ("transformer_256d_4head_3layer_weighted", "transformer", 3e-5, 16, 256, "MSE_PER_Weighted", 0.03, 4, 3),
]
import sys

# 명령행 인자로 실험 그룹과 GPU 받기
'''exp_group = int(sys.argv[1]) if len(sys.argv) > 1 else 1
gpu_id = int(sys.argv[2]) if len(sys.argv) > 2 else 6

# base_config의 GPU 설정 업데이트
base_config["training"]["device"] = f"cuda:{gpu_id}"

# 전체 실험을 반으로 나누기
total_experiments = len(experiments)
mid_point = total_experiments // 2

if exp_group == 1:
    experiments = experiments[:mid_point]  # 첫 번째 절반
    print(f"🚀 Running experiments 1-{mid_point} (Group 1) on GPU {gpu_id}")
else:
    experiments = experiments[mid_point:]  # 두 번째 절반
    print(f"🚀 Running experiments {mid_point+1}-{total_experiments} (Group 2) on GPU {gpu_id}")
    
'''
# 명령행 인자로 GPU만 받기 (그룹 분할 제거)
gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 6

# base_config의 GPU 설정 업데이트
base_config["training"]["device"] = f"cuda:{gpu_id}"

# 그룹 분할 코드 전체 주석처리 또는 삭제
# exp_group = int(sys.argv[1]) if len(sys.argv) > 1 else 1
# total_experiments = len(experiments)
# mid_point = total_experiments // 2
# if exp_group == 1:
#     experiments = experiments[:mid_point]
# else:
#     experiments = experiments[mid_point:]

# 전체 실험 실행
print(f"🚀 Running all {len(experiments)} experiments on GPU {gpu_id}")

print(f"🚀 Starting {len(experiments)} experiments...")
print(f"📋 실험 구성:")
print(f"  - Simple 모델: {sum(1 for exp in experiments if exp[1] == 'simple')}개")
print(f"  - Transformer 모델: {sum(1 for exp in experiments if exp[1] == 'transformer')}개")
print(f"  - 손실 함수별: MSE({sum(1 for exp in experiments if exp[4] == 'MSELoss')}), PER({sum(1 for exp in experiments if exp[4] == 'PredictionErrorRate')}), Combined({sum(1 for exp in experiments if exp[4] == 'MSE_PER_Weighted')})")

#for i, (exp_name, model_name, lr, batch, d_model, loss, decay) in enumerate(experiments, 1):
for i, (exp_name, model_name, lr, batch, d_model, loss, decay, nhead, num_layers) in enumerate(experiments, 1):
    print(f"\n{'='*60}")
    print(f"🚀 Experiment {i}/{len(experiments)}: {exp_name}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🤖 Model: {model_name}")
    print(f"⚙️  LR: {lr}, Batch: {batch}, d_model: {d_model}")
    print(f"📊 Loss: {loss}, Weight_decay: {decay}")
    print(f"{'='*60}")
    
    try:
        # 설정 복사 및 업데이트
        config = base_config.copy()
        config["model"]["name"] = model_name  # 👈 모델명 설정
        config["training"]["learning_rate"] = lr
        config["training"]["batch_size"] = batch
        config["model"]["d_model"] = d_model
        config["training"]["loss"] = loss
        config["training"]["optimizer_params"]["weight_decay"] = decay
        config["wandb"]["project"] = f"solar_prediction_comparison"
        config["experiment_name"] = exp_name
        config["model"]["nhead"] = nhead
        config["model"]["num_encoder_layers"] = num_layers
                
        # Transformer 모델 특별 설정
        if model_name == "transformer":
            config["model"]["num_encoder_layers"] = 2  # Transformer는 layer 2개
        else:
            config["model"]["num_encoder_layers"] = 1  # Simple은 의미없지만 유지
        
        # 설정 파일 저장
        config_path = f"solar_config_{exp_name}.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        # 실험 실행
        cfg = load_config(config_path)
        setup_wandb(cfg)
        model, best_metrics, test_metrics, external_test_metrics = train_solar_prediction_model(cfg)
        
        print(f"✅ Experiment {exp_name} SUCCESS!")
        print(f"📊 Best Val Metrics: {best_metrics}")
        print(f"📊 Test PredictionErrorRate: {test_metrics.get('PredictionErrorRate', 'N/A'):.1f}%")
        
        # 간단한 성능 기록 (나중에 비교용)
        with open("experiment_results.txt", "a") as f:
            val_per = best_metrics.get('PredictionErrorRate', 999) if best_metrics else 999
            test_per = test_metrics.get('PredictionErrorRate', 999) if test_metrics else 999
            f.write(f"{exp_name},{model_name},{d_model},{loss},{val_per:.1f},{test_per:.1f}\n")
        
    except Exception as e:
        print(f"❌ Experiment {exp_name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # 실패 기록
        with open("experiment_results.txt", "a") as f:
            f.write(f"{exp_name},{model_name},{d_model},{loss},FAILED,FAILED\n")
    
    finally:
        # GPU 메모리 정리
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 마지막 실험이 아니면 대기 (시간 조정)
        if i < len(experiments):
            wait_time = 20 if d_model <= 64 else 30  # 작은 모델은 짧게 대기
            print(f"😴 Waiting {wait_time} seconds for GPU cooldown...")
            import time
            time.sleep(wait_time)

print(f"\n🏆 All {len(experiments)} experiments completed!")

# 결과 요약 출력
try:
    print(f"\n📊 실험 결과 요약:")
    with open("experiment_results.txt", "r") as f:
        lines = f.readlines()
        for line in lines[-len(experiments):]:  # 방금 실행한 실험들만
            parts = line.strip().split(",")
            if len(parts) >= 6:
                exp_name, model_name, d_model, loss, val_per, test_per = parts[:6]
                print(f"  {exp_name:20s} | {model_name:11s} | d{d_model:3s} | {loss:15s} | Val: {val_per:>6s}% | Test: {test_per:>6s}%")
except:
    print("결과 요약 파일 읽기 실패")

print("\n🌞 Solar Power Prediction Training Completed! 🌞")