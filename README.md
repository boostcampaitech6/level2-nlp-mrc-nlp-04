

## yaml 파일 연결해서 사용하기! train_custom 사용방법
args.yaml 파일을 들어가서 확인하고 사용해 주세요!
갖다 붙인거라 사용법이 하드코딩이 많습니다..

1. model.saved_model_path , data.train_dataset_name , data.test_dataset_name , train.train_output_dir, train.inference_output_dir 이 절대 경로로 되어있으니 자기 컴퓨터에 맞게 바꿔주세요.

2. 특히 train_output_dir 의 경우 --overwrite_output_dir 이 안되니, /model/해당모델할파일이름 식으로 저장해주세요

3. python train_custom.py 로 실행시켜 학습을 시킵니다. eval 또한 EM 기준으로 중간중간 진행되며 wandb에 로깅됩니다. 로깅기준값은 args에서 변경 가능합니다

4. inference 의 경우 기존 방식과 똑같이 하되, 
python ./code/inference_bm25.py --output_dir ./outputs/test_dataset/ --dataset_name ./data/test_dataset/ """--model_name_or_path ./models/2/""" --overwrite_output_dir --do_predict 
--model_name_or_path 부분의 경로를 수정하여, 저장된 모델을 불러와 사용할수있도록합니다.

그 외에도 실행시 필요한 wandb, omagayaml 등 필요한 라이브러러리를 받아주세요
