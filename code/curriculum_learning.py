import os
import sys
# import yaml
import json
# import wandb
import logging
import evaluate
from typing import Dict
from evaluation import f1_score
from datasets import DatasetDict, load_from_disk
from transformers.trainer_callback import ProgressCallback
from utils_qa import check_no_error, postprocess_qa_predictions
from trainer_qa import QuestionAnsweringTrainer, CustomProgressCallback
from arguments import DataTrainingArguments, ModelArguments, CustomTrainingArguments
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    EarlyStoppingCallback,
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForSeq2SeqLM,
)
from utils_qa import check_no_error, postprocess_qa_predictions, set_seed
from utils.logging_utils import setup_logging
from utils.file_name_utils import save_custom_metrics

seed = 2024
logger = logging.getLogger(__name__)

def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args.model_name_or_path)
    
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    setup_logging()

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)
    '''
    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    '''


def get_column_names(training_args, datasets):
    if training_args.do_train:
        return datasets["train"].column_names
    else:
        return datasets["validation"].column_names


def get_tokenized_examples_start_end_positions(tokenized_examples_origin, offset_mapping, tokenizer, sample_mapping, examples, answer_column_name,
                                               pad_on_right):
    tokenized_examples = tokenized_examples_origin

    # 데이터셋에 "start position", "end position" label을 부여합니다.
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

        # sequence id를 설정합니다 (to know what is the context and what is the question).
        # token_id와 다른 점은 special token 값이 None이라는 점입니다.
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]

        # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
        # chunk가 여러 개일 때, 답이 없는 chunk가 있을 수 있기 때문
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # answer의 후보군이 여러 개일 경우 첫 번째 값을 가져옵니다.
            # text에서 정답의 Start/end character index
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # text에서 current span의 Start token index
            # Start token index은 앞에서부터 index를 찾습니다.
            token_start_index = 0
            # query, special token 제외
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # text에서 current span의 End token index
            # End token index은 뒤에서부터 index를 찾습니다.
            token_end_index = len(input_ids) - 1
            # query, special token 제외
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                while (token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def get_tokenized_examples_example_id(tokenized_examples_origin, pad_on_right, sample_mapping, examples):
    tokenized_examples = tokenized_examples_origin
    # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
    # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # sequence id를 설정합니다 (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # 하나의 example이 여러개의 span을 가질 수 있습니다.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None) for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples


def check_validation_datasets_error(datasets):
    if "train" not in datasets:
        raise ValueError("--do_train requires a train dataset")


def formatting_for_metric(predictions, training_args, answer_column_name, datasets):
    # Metric을 구할 수 있도록 Format을 맞춰줍니다.
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    if training_args.do_predict:
        return formatted_predictions

    elif training_args.do_eval:
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in datasets["validation"]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def get_train_dataset(datasets, prepare_train_features, data_args, column_names):
    check_validation_datasets_error(datasets)

    train_dataset = datasets["train"]

    # dataset에서 train feature를 생성합니다.
    train_dataset = train_dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    return train_dataset


def get_eval_dataset(datasets, prepare_validation_features, data_args, column_names):
    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    return eval_dataset


def do_training(training_args, last_checkpoint, model_args, trainer, train_dataset):
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    elif os.path.isdir(model_args.model_name_or_path):
        checkpoint = model_args.model_name_or_path
    else:
        checkpoint = None

    # logging_step마다 콘솔에 로그를 출력하지 않도록 수정
    # TODO: ProgressCallback stdout 개선방향 찾아보기
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(CustomProgressCallback)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # State 저장
    trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))


def do_evaluation(trainer, eval_dataset):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: CustomTrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> None:

    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    column_names = get_column_names(training_args, datasets)

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding에 대한 옵션을 설정합니다.
    # (question|context) 혹은 (context|question)로 세팅 가능합니다.
    # pad_on_right
    #    True: query + context
    #    False: context + query
    # TODO: tokenizer padding side setting
    pad_on_right = tokenizer.padding_side == "right"

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(data_args, training_args, datasets, tokenizer)

    # Train preprocessing / 전처리를 진행합니다.
    def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        # max seq length를 넘어가면 여러 개의 chunk로 나눠집니다.
        # ex) 두 개의 context가 있을 때,
        # 첫 번째의 query+context는 max seq length를 넘어가지 않았고,
        # 두 번째의 query+context는 max seq length를 넘어간다면
        # sample_mapping이 [0,1,1, ...]가 됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False if model.base_model_prefix == "roberta" else True,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        # offset_mapping: List[(단어의 start index, 단어의 end index+1)]
        # 이때, [SEP]와 [CLS] 토큰 같은 special token의 값은 (0,0)으로 mapping 됩니다. (tokenizer에서 자동으로 처리)
        offset_mapping = tokenized_examples.pop("offset_mapping")

        tokenized_examples = get_tokenized_examples_start_end_positions(tokenized_examples, offset_mapping, tokenizer, sample_mapping, examples,
                                                                        answer_column_name, pad_on_right)

        return tokenized_examples

    if training_args.do_train:
        train_dataset = get_train_dataset(datasets, prepare_train_features, data_args, column_names)

    # Validation preprocessing
    # validation은 train과 달리, chunk에서 start/end token의 위치를 알아낼 필요가 없습니다.
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_token_type_ids=False if model.base_model_prefix == "roberta" else True,  # roberta모델을 사용할 경우 False, bert를 사용할 경우 True
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        tokenized_examples = get_tokenized_examples_example_id(tokenized_examples, pad_on_right, sample_mapping, examples)

        return tokenized_examples

    if training_args.do_eval:
        eval_dataset = get_eval_dataset(datasets, prepare_validation_features, data_args, column_names)

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    # Question: pad_to_multiple_of=8, training_args.fp16
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )

        return formatting_for_metric(predictions, training_args, answer_column_name, datasets)

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Early Stopping Callback
    # callbacks = None
    # if training_args.early_stopping:
    #     early_stop = EarlyStoppingCallback(training_args.early_stopping)
    #     callbacks = [early_stop]

    def cal_f1_score(predictions, examples):
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        references = {ex["id"]: ex["answers"] for ex in examples}
        f1_pred = []
        for id in references:
            predicted = predictions[id]
            groundtruth = references[id]["text"][0]
            f1 = f1_score(predicted, groundtruth)
            f1_pred.append(f1)

        examples = examples.add_column('f1', f1_pred)
        examples = examples.sort('f1', reverse=True)

        return examples

    # Curriculum Learning
    if training_args.do_train:
        with open(data_args.curriculum_learning_prediction_file, 'r', encoding='UTF-8') as f:
            # key: id, value: f1 score
            prediction: Dict[int, float] = json.load(f)

        sorted_by_f1 = cal_f1_score(prediction, datasets["train"])

        num_subsets = len(sorted_by_f1) // 5
        data_subsets = []

        for i in range(0, len(sorted_by_f1), num_subsets):
            if i < 4 * num_subsets:
                data_subsets.append(sorted_by_f1.select(range(i, i + num_subsets)))
            else:
                data_subsets.append(sorted_by_f1.select(range(i, len(sorted_by_f1))))
                break

        feature_subsets = []
        for subset in data_subsets:
            subset = subset.map(
                prepare_train_features,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=subset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            feature_subsets.append(subset)

        for subset_data in feature_subsets:

            # Trainer 초기화
            trainer = QuestionAnsweringTrainer(model=model,
                                               args=training_args,
                                               train_dataset=subset_data if training_args.do_train else None,
                                               eval_dataset=eval_dataset if training_args.do_eval else None,
                                               eval_examples=datasets["validation"] if training_args.do_eval else None,
                                               tokenizer=tokenizer,
                                               data_collator=data_collator,
                                               post_process_function=post_processing_function,
                                               compute_metrics=compute_metrics,
                                            #    callbacks=callbacks
                                               )

            do_training(training_args, last_checkpoint, model_args, trainer, subset_data)

    # Evaluation
    if training_args.do_eval:
        # Trainer 초기화
        trainer = QuestionAnsweringTrainer(model=model,
                                           args=training_args,
                                           train_dataset=train_dataset if training_args.do_train else None,
                                           eval_dataset=eval_dataset if training_args.do_eval else None,
                                           eval_examples=datasets["validation"] if training_args.do_eval else None,
                                           tokenizer=tokenizer,
                                           data_collator=data_collator,
                                           post_process_function=post_processing_function,
                                           compute_metrics=compute_metrics,
                                        #    callbacks=callbacks
                                           )
        do_evaluation(trainer, eval_dataset)


if __name__ == "__main__":
    main()