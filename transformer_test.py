import torch.nn as nn

# 설정값
d_model = 512  # 단어를 표현하는 벡터의 크기
nhead = 8      # Attention Head의 개수

# Transformer의 한 층(Layer) 정의
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

print(encoder_layer)
print(encoder_layer.self_attn.in_proj_weight.shape)

# 입력 데이터 (문장 길이 10, 배치 크기 32, 벡터 크기 512)
src = torch.rand(10, 32, 512)
out = encoder_layer(src)

print(out.shape) # 결과도 입력과 같은 크기로 나옵니다.