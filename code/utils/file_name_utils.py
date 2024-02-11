import os
import json

from .datetime_helper import get_seoul_datetime_str


def save_custom_metrics(trainer, metrics, prefix='eval'):
    # 날짜와 시간 문자열 가져오기
    date_time_str = get_seoul_datetime_str()
    # 사용자 정의 파일 이름 구성
    scores = '_'.join([f"{v:.4f}" for k, v in metrics.items()])

    custom_filename = f"{prefix}_{date_time_str}_{scores}.json"
    
    # 평가 결과 저장 디렉토리
    save_path = os.path.join(trainer.args.output_dir, custom_filename)
    
    # 메트릭을 사용자 정의 파일에 저장
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {save_path}")
