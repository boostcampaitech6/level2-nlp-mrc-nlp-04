#!/bin/bash
if [ "$(id -u)" -eq 0 ]; then
  chmod +x train.sh eval.sh inference.sh
fi

# 학습 스크립트 실행
echo "Starting training..."
./train.sh
echo "Training completed."

# 평가 스크립트 실행
echo "Starting evaluation..."
./eval.sh
echo "Evaluation completed."

# 추론 스크립트 실행
echo "Starting inference..."
./inference.sh
echo "Inference completed."
