import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class SimpleCNNEncoder(nn.Module):
    """
    CNN 기반 Historical Encoder
    Transformer보다 가볍고 패딩도 자연스럽게 처리
    """
    def __init__(self, input_dim: int = 54, hidden_dim: int = 128):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 1D CNN layers for temporal patterns
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(3)
        ])
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, 54]
            padding_mask: [batch, seq_len] - True for padded positions
        Returns:
            encoded: [batch, seq_len, hidden_dim]
        """
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, hidden_dim]
        
        # Apply padding mask if provided
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        # CNN processing
        x_conv = x.transpose(1, 2)  # [batch, hidden_dim, seq_len] for conv1d
        
        for conv, norm in zip(self.conv_layers, self.layer_norms):
            # Convolution
            conv_out = conv(x_conv)  # [batch, hidden_dim, seq_len]
            conv_out = F.gelu(conv_out)
            
            # Residual connection + layer norm
            x_conv = x_conv + self.dropout(conv_out)
            x_conv = norm(x_conv.transpose(1, 2)).transpose(1, 2)  # Normalize over hidden_dim
        
        encoded = x_conv.transpose(1, 2)  # [batch, seq_len, hidden_dim]
        
        # Apply padding mask again
        if padding_mask is not None:
            encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        
        return encoded


class SimpleAttentionPooling(nn.Module):
    """
    간단한 Attention으로 24시간 context 생성
    각 예측 시간별로 historical sequence에서 중요한 정보 추출
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 24시간 각각에 대한 쿼리 벡터
        self.hour_queries = nn.Parameter(torch.randn(24, hidden_dim) * 0.1)
        
        # Simple attention
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, encoded_hist: torch.Tensor, 
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            encoded_hist: [batch, seq_len, hidden_dim]
            padding_mask: [batch, seq_len]
        Returns:
            hourly_contexts: [batch, 24, hidden_dim]
        """
        batch_size = encoded_hist.size(0)
        
        # 24시간 쿼리 확장
        queries = self.hour_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 24, hidden_dim]
        
        # Q, K, V projection
        Q = self.query_proj(queries)  # [batch, 24, hidden_dim]
        K = self.key_proj(encoded_hist)  # [batch, seq_len, hidden_dim]
        V = self.value_proj(encoded_hist)  # [batch, seq_len, hidden_dim]
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch, 24, seq_len]
        
        # Apply padding mask
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask.unsqueeze(1), float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch, 24, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum
        hourly_contexts = torch.matmul(attn_weights, V)  # [batch, 24, hidden_dim]
        
        return hourly_contexts


class SimplePredictionHead(nn.Module):
    """
    간단한 MLP 기반 예측 헤드
    Historical context + Weather + Plant meta → 24시간 예측
    """
    def __init__(self, historical_dim: int = 128, weather_dim: int = 5, plant_dim: int = 10):
        super().__init__()
        
        # Weather와 Plant meta 처리
        self.weather_encoder = nn.Sequential(
            nn.Linear(weather_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1)
        )
        
        self.plant_encoder = nn.Sequential(
            nn.Linear(plant_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32)
        )
        
        # 융합된 특성 처리
        fused_dim = historical_dim + 32 + 32  # 192
        
        # 간단한 MLP
        self.prediction_mlp = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.GELU(),
            
            nn.Linear(32, 1)  # 각 시간별로 1개 값 예측
        )
        
    def forward(self, hourly_contexts: torch.Tensor, 
                weather_data: torch.Tensor, plant_meta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hourly_contexts: [batch, 24, hidden_dim]
            weather_data: [batch, 24, weather_dim]
            plant_meta: [batch, plant_dim]
        Returns:
            predictions: [batch, 24]
        """
        batch_size = hourly_contexts.size(0)
        
        # Weather encoding
        weather_encoded = self.weather_encoder(weather_data)  # [batch, 24, 32]
        
        # Plant meta encoding and expansion
        plant_encoded = self.plant_encoder(plant_meta)  # [batch, 32]
        plant_expanded = plant_encoded.unsqueeze(1).expand(-1, 24, -1)  # [batch, 24, 32]
        
        # Feature fusion
        fused_features = torch.cat([
            hourly_contexts,    # [batch, 24, hidden_dim]
            weather_encoded,    # [batch, 24, 32]
            plant_expanded      # [batch, 24, 32]
        ], dim=-1)  # [batch, 24, fused_dim]
        
        # Prediction for each hour
        predictions = self.prediction_mlp(fused_features).squeeze(-1)  # [batch, 24]
        
        return predictions


class SimpleSolarPredictor(nn.Module):
    """
    간단한 태양광 발전량 예측 모델
    
    Architecture:
    1. CNN Encoder - 시계열 패턴 추출
    2. Simple Attention - 24시간별 context 생성
    3. MLP Prediction - 최종 예측
    """
    
    def __init__(
        self,
        historical_features: int = 54,
        weather_features: int = 5,
        plant_features: int = 10,
        hidden_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.device = device
        self.hidden_dim = hidden_dim
        
        # 1. CNN-based Historical Encoder
        self.historical_encoder = SimpleCNNEncoder(
            input_dim=historical_features,
            hidden_dim=hidden_dim
        )
        
        # 2. Simple Attention Pooling
        self.attention_pooling = SimpleAttentionPooling(
            hidden_dim=hidden_dim
        )
        
        # 3. Simple Prediction Head
        self.prediction_head = SimplePredictionHead(
            historical_dim=hidden_dim,
            weather_dim=weather_features,
            plant_dim=plant_features
        )
        
        # 4. Output activation
        self.output_activation = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)  # 작은 gain으로 안정적 초기화
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(
        self,
        historical_data: torch.Tensor,
        weather_data: torch.Tensor,
        plant_meta: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            historical_data: [batch, seq_len, 54]
            weather_data: [batch, 24, 5]
            plant_meta: [batch, 10] -> will use first 7 features
            padding_mask: [batch, seq_len]
            
        Returns:
            Dict containing predictions and intermediate outputs
        """
        
        # Plant meta에서 첫 7개 특성만 사용
        #plant_meta_used = plant_meta[:, :7]
        
        # 1. Historical encoding with CNN
        encoded_hist = self.historical_encoder(historical_data, padding_mask)
        # [batch, seq_len, hidden_dim]
        
        # 2. 24시간별 context 생성
        hourly_contexts = self.attention_pooling(encoded_hist, padding_mask)
        # [batch, 24, hidden_dim]
        
        # 3. 최종 예측
        predictions = self.prediction_head(hourly_contexts, weather_data, plant_meta)
        # [batch, 24]
        
        # 4. 후처리 (발전량은 항상 양수)
        predictions = self.output_activation(predictions)
        
        outputs = {
            'predictions': predictions,
            'hourly_contexts': hourly_contexts,
            'encoded_hist': encoded_hist
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
            'model_name': 'SimpleSolarPredictor',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / 1024 / 1024,  # float32 기준
            'hidden_dim': self.hidden_dim,
            'device': self.device,
            'architecture': 'CNN + Simple Attention + MLP'
        }


def create_simple_solar_model(
    historical_features: int = 54,
    weather_features: int = 5,
    plant_features: int = 10,
    hidden_dim: int = 128,
    device: str = "auto"
) -> SimpleSolarPredictor:
    """
    간단한 태양광 발전량 예측 모델 생성
    
    Args:
        historical_features: Historical 데이터 특성 수 (기본 54)
        weather_features: Weather 데이터 특성 수 (기본 5)  
        plant_features: Plant meta 데이터 특성 수 (기본 7)
        hidden_dim: 모델 히든 차원 (기본 128)
        device: 디바이스 설정
        
    Returns:
        SimpleSolarPredictor 모델 인스턴스
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = SimpleSolarPredictor(
        historical_features=historical_features,
        weather_features=weather_features,
        plant_features=plant_features,
        hidden_dim=hidden_dim,
        device=device
    )
    
    model = model.to(device)
    
    print("🌞 Simple Solar Power Prediction Model Created! 🌞")
    print("=" * 55)
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")
    print("=" * 55)
    
    return model

'''
# ===== 테스트 코드 =====
if __name__ == "__main__":
    print("Creating Simple Solar Power Prediction Model...")
    model = create_simple_solar_model(hidden_dim=64, device="cuda" if torch.cuda.is_available() else "cpu")
    
    # 테스트 데이터 생성
    batch_size = 4
    seq_len = 48
    device = model.device
    
    print(f"\nGenerating test data on device: {device}")
    
    historical_data = torch.randn(batch_size, seq_len, 54).to(device)
    weather_data = torch.randn(batch_size, 24, 5).to(device)
    plant_meta = torch.randn(batch_size, 10).to(device)  # 10개 특성 중 7개만 사용
    
    # 패딩 마스크
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)
    padding_mask[:2, -10:] = True  # 일부 샘플에 패딩
    
    print(f"Test data shapes:")
    print(f"  Historical: {historical_data.shape}")
    print(f"  Weather: {weather_data.shape}")
    print(f"  Plant meta: {plant_meta.shape}")
    
    # Forward pass 테스트
    print(f"\nTesting forward pass...")
    model.eval()
    
    with torch.no_grad():
        outputs = model(historical_data, weather_data, plant_meta, padding_mask)
    
    predictions = outputs['predictions']
    print(f"✅ Forward pass successful!")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[0][:6].cpu().numpy()}")
    
    # 추론 테스트
    predictions_infer = model.predict(historical_data, weather_data, plant_meta, padding_mask)
    print(f"✅ Inference successful!")
    
    # 메모리 사용량
    if torch.cuda.is_available():
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    
    print(f"\n🎉 Simple model tests passed! 🎉")
'''