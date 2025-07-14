import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple


class PaddingAwareEncoder(nn.Module):
    """
    패딩을 인식하는 Historical Encoder
    시간 순서를 보존하면서 모든 54개 features 처리
    """
    def __init__(self, input_dim: int = 54, d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=48)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, 54] - historical data
            padding_mask: [batch, seq_len] - True for padded positions
            
        Returns:
            encoded: [batch, seq_len, d_model] - 시간 정보 보존하며 인코딩
        """
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Scale by sqrt(d_model) as in original transformer
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer encoding with padding mask
        encoded = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Layer normalization
        encoded = self.layer_norm(encoded)
        
        return encoded


class PositionalEncoding(nn.Module):
    """표준 positional encoding"""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class HourlyAttentionFusion(nn.Module):
    """
    24시간 각 시간별로 historical sequence에서 관련 정보를 선택적으로 추출
    """
    def __init__(self, d_model: int = 256, nhead: int = 8):
        super().__init__()
        
        self.d_model = d_model
        
        # 24시간 각각에 대한 학습 가능한 쿼리 벡터
        self.hour_queries = nn.Parameter(torch.randn(24, d_model) / math.sqrt(d_model))
        
        # Cross attention: 각 예측 시간 → historical sequence
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True
        )
        
        # Context refinement
        self.context_refine = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, historical_encoded: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            historical_encoded: [batch, seq_len, d_model] - 인코딩된 historical sequence
            padding_mask: [batch, seq_len] - 패딩 마스크
            
        Returns:
            hourly_contexts: [batch, 24, d_model] - 각 시간별 context
        """
        batch_size = historical_encoded.size(0)
        
        # 24시간 쿼리를 배치 크기만큼 복제
        queries = self.hour_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 24, d_model]
        
        # Cross attention: 각 예측 시간이 historical sequence에서 관련 정보 추출
        hourly_contexts, attention_weights = self.cross_attention(
            query=queries,                    # [batch, 24, d_model] - 24시간 쿼리
            key=historical_encoded,           # [batch, seq_len, d_model] - historical
            value=historical_encoded,         # [batch, seq_len, d_model] - historical
            key_padding_mask=padding_mask     # [batch, seq_len] - 패딩 마스크
        )
        
        # Context 정제
        hourly_contexts = self.context_refine(hourly_contexts)  # [batch, 24, d_model]
        
        return hourly_contexts


class PatchMixerBlock(nn.Module):
    """PatchTSMixer의 핵심 MLP Mixer 블록"""
    def __init__(self, dim: int, expansion_factor: int = 4):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Feature mixing MLP
        self.feature_mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(0.1)
        )
        
        # Residual scaling
        self.gamma1 = nn.Parameter(torch.ones(1))
        self.gamma2 = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, dim]
        Returns:
            x: [batch, seq_len, dim]
        """
        # Feature mixing with residual connection
        residual = x
        x = self.norm1(x)
        x = residual + self.gamma1 * self.feature_mlp(x)
        
        # Final normalization
        x = self.norm2(x)
        
        return x


class ChannelMixer(nn.Module):
    """특성 간 상호작용 (Channel mixing)"""
    def __init__(self, dim: int):
        super().__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 24, dim] - 각 시간별로 특성 mixing
        """
        return self.channel_mlp(self.norm(x))


class TemporalMixer(nn.Module):
    """시간 간 상호작용 (Temporal mixing)"""
    def __init__(self, seq_len: int = 24):
        super().__init__()
        
        self.norm = nn.LayerNorm(seq_len)
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(seq_len * 2, seq_len),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 24, dim]
        Returns:
            x: [batch, 24, dim]
        """
        # [batch, 24, dim] → [batch, dim, 24] → mixing → [batch, dim, 24] → [batch, 24, dim]
        x_t = x.transpose(1, 2)  # [batch, dim, 24]
        x_t_mixed = self.temporal_mlp(self.norm(x_t))  # [batch, dim, 24]
        return x_t_mixed.transpose(1, 2)  # [batch, 24, dim]


class OptimalPatchTSMixerFusion(nn.Module):
    """
    PatchTSMixer 기반 최종 예측 모듈
    Historical context + Weather + Plant meta → 24시간 발전량 예측
    """
    def __init__(self, historical_dim: int = 256, weather_dim: int = 5, plant_dim: int = 7):
        super().__init__()
        
        # 모달리티별 projection
        self.weather_proj = nn.Sequential(
            nn.Linear(weather_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        self.plant_proj = nn.Sequential(
            nn.Linear(plant_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        # 융합된 특성 차원
        self.fused_dim = historical_dim + 32 + 32  # 320
        
        # PatchTSMixer 스타일 블록들
        self.mixer_blocks = nn.ModuleList([
            PatchMixerBlock(self.fused_dim) for _ in range(3)
        ])
        
        # Channel mixing (특성 간 상호작용)
        self.channel_mixer = ChannelMixer(self.fused_dim)
        
        # Temporal mixing (24시간 간 상호작용)  
        self.temporal_mixer = TemporalMixer(24)
        
        # 중간 projection
        self.mid_projection = nn.Sequential(
            nn.Linear(self.fused_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1)
        )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, hourly_historical: torch.Tensor, weather_data: torch.Tensor, 
                plant_meta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hourly_historical: [batch, 24, 256] - 시간별 historical context
            weather_data: [batch, 24, 5] - 기상 데이터
            plant_meta: [batch, 7] - 발전소 메타
            
        Returns:
            predictions: [batch, 24] - 24시간 발전량 예측
        """
        batch_size = hourly_historical.size(0)
        
        # 1. 모달리티 융합
        weather_embed = self.weather_proj(weather_data)  # [batch, 24, 32]
        plant_embed = self.plant_proj(plant_meta)  # [batch, 32]
        plant_expanded = plant_embed.unsqueeze(1).expand(-1, 24, -1)  # [batch, 24, 32]
        
        # 모든 정보 결합
        fused_features = torch.cat([
            hourly_historical,  # [batch, 24, 256] - 시간별 historical context
            weather_embed,      # [batch, 24, 32] - 시간별 기상 정보
            plant_expanded      # [batch, 24, 32] - 발전소 특성
        ], dim=-1)  # [batch, 24, 320]
        
        # 2. PatchTSMixer 스타일 처리
        x = fused_features
        
        # 여러 Mixer 블록 통과
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x) + x  # Residual connection -> 중복인듯
            #x = mixer_block(x)
            
        # 3. Channel mixing (특성 간 상호작용)
        x = x + self.channel_mixer(x)  # Residual connection
        
        # 4. Temporal mixing (시간 간 상호작용)
        x = x + self.temporal_mixer(x)  # Residual connection
        
        # 5. 중간 projection
        x = self.mid_projection(x)  # [batch, 24, 128]
        
        # 6. 최종 예측
        predictions = self.prediction_head(x).squeeze(-1)  # [batch, 24]
        
        return predictions


class SolarPowerPredictor(nn.Module):
    """
    최종 태양광 발전량 예측 모델
    
    Architecture:
    1. Historical Encoder (Transformer) - 시간 정보 보존
    2. Hourly Attention - 각 예측 시간별 context 추출  
    3. PatchTSMixer Fusion - 최종 예측
    """
    
    def __init__(
        self,
        historical_features: int = 54,
        weather_features: int = 5,
        plant_features: int = 7,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.d_model = d_model
        
        # 1. Historical Encoder (시간 정보 보존)
        self.historical_encoder = PaddingAwareEncoder(
            input_dim=historical_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers
        )
        
        # 2. 시간별 Attention (각 예측 시간별 context)
        self.hourly_attention = HourlyAttentionFusion(
            d_model=d_model,
            nhead=nhead
        )
        
        # 3. PatchTSMixer 기반 Final Prediction
        self.final_predictor = OptimalPatchTSMixerFusion(
            historical_dim=d_model,
            weather_dim=weather_features,
            plant_dim=plant_features
        )
        
        # 4. 출력 처리
        self.output_activation = nn.ReLU()
        
        # 모델 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
    def forward(
        self,
        historical_data: torch.Tensor,
        weather_data: torch.Tensor,
        plant_meta: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            historical_data: [batch, seq_len, 54] - 이틀간 historical 데이터 (seq_len=48)
            weather_data: [batch, 24, 5] - 기상 예보  
            plant_meta: [batch, 10] - 발전소 메타데이터 (10개 특성)
            padding_mask: [batch, seq_len] - 패딩 마스크
            return_attention: attention weights 반환 여부
            
        Returns:
            Dict containing predictions and intermediate outputs
        """
        
        # 1. Historical encoding (시간 정보 보존)
        historical_encoded = self.historical_encoder(historical_data, padding_mask)
        # [batch, seq_len, d_model] - 모든 시간 정보 유지!
        
        # 2. 24시간 각각에 대한 selective attention
        hourly_contexts = self.hourly_attention(historical_encoded, padding_mask)
        # [batch, 24, d_model] - 각 예측 시간별 최적 context
        
        # 3. PatchTSMixer 기반 최종 예측
        predictions = self.final_predictor(hourly_contexts, weather_data, plant_meta)
        # [batch, 24] - 시계열 예측 전문 모델로 처리
        
        # 4. 후처리 (발전량은 항상 양수)
        predictions = self.output_activation(predictions)
        
        outputs = {
            'predictions': predictions,
            'hourly_contexts': hourly_contexts,
            'historical_encoded': historical_encoded
        }
        
        return outputs
        
    def predict(
        self, 
        historical_data: torch.Tensor, 
        weather_data: torch.Tensor,
        plant_meta: torch.Tensor, 
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """추론용 함수"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(historical_data, weather_data, plant_meta, padding_mask)
            return outputs['predictions']
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'SolarPowerPredictor',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # float32 기준
            'preserves_temporal_info': True,
            'handles_padding': True,
            'uses_all_features': True,
            'historical_features': 54,
            'd_model': self.d_model,
            'device': self.device,
            'architecture': 'Transformer + PatchTSMixer Fusion'
        }


def create_solar_prediction_model(
    historical_features: int = 54,
    weather_features: int = 5,
    plant_features: int = 7,
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 4,

    device: str = "auto"
) -> SolarPowerPredictor:
    """
    태양광 발전량 예측 모델 생성
    
    Args:
        historical_features: Historical 데이터 특성 수 (기본 54)
        weather_features: Weather 데이터 특성 수 (기본 5)  
        plant_features: Plant meta 데이터 특성 수 (기본 7)
        d_model: Transformer 모델 차원 (기본 256)
        device: 디바이스 설정
        
    Returns:
        SolarPowerPredictor 모델 인스턴스
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SolarPowerPredictor(
        historical_features=historical_features,
        weather_features=weather_features,
        plant_features=plant_features,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        device=device
    )
    
    model = model.to(device)
    
    print("🌞 Solar Power Prediction Model Created! 🌞")
    print("=" * 50)
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print("=" * 50)
    
    return model

'''
# 사용 예시 및 테스트

# 모델 생성
print("Creating Solar Power Prediction Model...")
model = create_solar_prediction_model(device="cuda" if torch.cuda.is_available() else "cpu")

# 테스트 데이터 생성
batch_size = 8
seq_len = 48    # 4일 * 24시간
device = model.device

print(f"\nGenerating test data on device: {device}")

# 실제 데이터 크기와 동일한 더미 데이터
historical_data = torch.randn(batch_size, seq_len, 54).to(device)
weather_data = torch.randn(batch_size, 24, 5).to(device)
plant_meta = torch.randn(batch_size, 7).to(device)

# 패딩 마스크 (14시/20시 시나리오 시뮬레이션)
padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
# 예시: 절반은 14시 시나리오, 절반은 20시 시나리오
padding_mask[:batch_size//2, -10:] = True  # 14시 시나리오 (10시간 패딩)
padding_mask[batch_size//2:, -4:] = True   # 20시 시나리오 (4시간 패딩)

print(f"Test data shapes:")
print(f"  Historical: {historical_data.shape}")
print(f"  Weather: {weather_data.shape}")
print(f"  Plant meta: {plant_meta.shape}")
print(f"  Padding mask: {padding_mask.shape}")

# Forward pass 테스트
print(f"\nTesting forward pass...")

model.eval()
with torch.no_grad():
    outputs = model(historical_data, weather_data, plant_meta, padding_mask)

predictions = outputs['predictions']
hourly_contexts = outputs['hourly_contexts']

print(f"✅ Forward pass successful!")
print(f"Predictions shape: {predictions.shape}")
print(f"Hourly contexts shape: {hourly_contexts.shape}")
print(f"Sample predictions (first sample): {predictions[0][:6].tolist()}")

# 추론 테스트
print(f"\nTesting inference...")
predictions_infer = model.predict(historical_data, weather_data, plant_meta, padding_mask)
print(f"✅ Inference successful!")
print(f"Inference predictions shape: {predictions_infer.shape}")

# 결과 일치 확인
assert torch.allclose(predictions, predictions_infer, atol=1e-6)
print(f"✅ Forward and inference results match!")

print(f"\n🎉 All tests passed! Model is ready for training! 🎉")

# 메모리 사용량 체크
if torch.cuda.is_available():
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    print(f"GPU memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.1f} MB")
'''