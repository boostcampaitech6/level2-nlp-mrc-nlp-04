import os
import json
import random
import time
import pickle
from contextlib import contextmanager
from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
from pyprnt import prnt
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig


from utils_qa import set_seed

seed = 2024
set_seed(seed)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class DenseRetrieval:

    def __init__(self, args, num_neg, tokenizer, p_encoder, q_encoder, num_sample: int = -1, data_path: Optional[str] = './data', context_path: Optional[str] = "wikipedia_documents.json",):

        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''
        self.args = args
        self.data_path = data_path
        self.dataset = load_from_disk(os.path.join(self.data_path, 'train_dataset'))
        self.train_dataset = self.dataset['train'] if num_sample is -1 else self.dataset['train'][:num_sample]
        self.valid_dataset = self.dataset['validation']
        testdata = load_from_disk(os.path.join(self.data_path), 'test_dataset')
        self.test_dataset = testdata['validation']
        del testdata

        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))
        
        # self.train_dataset = dataset[:num_sample]
        # self.valid_dataset = dataset[num_sample]
        self.num_neg = num_neg
        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.pwd = os.getcwd()
        self.save_path = os.path.join(self.pwd, 'models/dpr')

        self.prepare_in_batch_negative(num_neg=num_neg)


    def prepare_in_batch_negative(self, dataset=None, num_neg=3, tokenizer=None):

        if dataset is None:
            dataset = self.dataset
            dataset = concatenate_datasets([dataset["train"].flatten_indices(),
                                            dataset["validation"].flatten_indices()])

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in dataset['context']])))
        p_with_neg = []

        for c in dataset['context']:
            
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )
        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)


    def prepare_in_batch_negative_for_checkpoint(self):
        # reference: https://arxiv.org/abs/2012.12624
        # reference: https://github.com/boostcampaitech5/level2_nlp_mrc-nlp-04/blob/main/dpr_retrieval.py

        q_seqs = self.tokenizer(
            self.train_dataset['question'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        p_seqs = self.tokenizer(
            self.train_dataset['context'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )
        self.train_dataloader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size, drop_last=False)
        ###################################
        valid_q_seqs = self.tokenizer(
            self.valid_dataset['question'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        valid_dataset = TensorDataset(
            valid_q_seqs['input_ids'], valid_q_seqs['attention_mask'], valid_q_seqs['token_type_ids']
        )
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=False)
        ###################################
        test_q_seqs = self.tokenizer(
            self.test_dataset['question'], padding="max_length",
            truncation=True, return_tensors='pt'
        )
        test_dataset = TensorDataset(
            test_q_seqs['input_ids'], test_q_seqs['attention_mask'], test_q_seqs['token_type_ids']
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=False)
        ###################################
        wiki_seqs = self.tokenizer(
            self.contexts, padding="max_length",
            truncation=True, return_tensors='pt'
        )
        wiki_dataset = TensorDataset(
            wiki_seqs['input_ids'], wiki_seqs['attention_mask'], wiki_seqs['token_type_ids']
        )
        self.wiki_dataloader = DataLoader(
            wiki_dataset, batch_size=self.args.per_device_train_batch_size, drop_last=False)


    def train(self, args=None, override=False, num_pre_batch=0):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        for _ in train_iterator:
            p_queue = deque(maxlen=num_pre_batch)
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    p_encoder.train()
                    q_encoder.train()
            
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    if num_pre_batch is not 0:  # Pre-batch
                        p_inputs = {
                            'input_ids': batch[0].to(args.device),
                            'attention_mask': batch[1].to(args.device),
                            'token_type_ids': batch[2].to(args.device)
                        }
                    
                    else:   # In-batch negtive sampling
                        p_inputs = {
                            'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                            'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                            'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                        }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)
                    if num_pre_batch is not 0:  # Pre-batch negative sampling
                        temp = p_outputs.clone().detach()
                        p_outputs = torch.cat((p_outputs, *p_queue), dim=0)
                        p_queue.append(temp)

                        # Calculate similarity score & loss
                        sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1)).squeeze()
                        sim_scores = F.log_softmax(sim_scores, dim=1)
                        loss = F.nll_loss(sim_scores, targets)
                        tepoch.set_postfix(loss=f'{str(loss.item())}')

                    else:   # In-batch negative sampling
                        # Calculate similarity score & loss
                        p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                        q_outputs = q_outputs.view(batch_size, 1, -1)

                        sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                        sim_scores = sim_scores.view(batch_size, -1)
                        sim_scores = F.log_softmax(sim_scores, dim=1)

                        loss = F.nll_loss(sim_scores, targets)
                        tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()
                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(self.p_encoder.state_dict(), os.path.join(self.save_path, 'p_encoder_state_dict'))
        torch.save(self.q_encoder.state_dict(), os.path.join(self.save_path, 'q_encoder_state_dict'))
        print('encoder statedict saved')


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk, p_encoder=self.p_encoder, q_encoder=self.q_encoder
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas


    def get_q_embs(self):
        with torch.no_grad():
            self.q_encoder.eval()

            self.q_embs = []
            for batch in tqdm(self.passage_dataloader):
                q_inputs = {
                    'input_ids': batch[0].to(self.args.device),
                    'attention_mask': batch[1].to(self.args.device),
                    'token_type_ids': batch[2].to(self.args.device)
                }
                q_emb = self.q_encoder(**q_inputs).to('cpu')
                self.q_embs.append(q_emb)
        return self.q_embs


    def get_relevant_doc(self, query, k=10, args=None, p_encoder=None, q_encoder=None):

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        doc_score = dot_prod_scores.squeeze()[rank][:k]
        doc_indices = rank.tolist()[:k]
        return doc_score, doc_indices


    def get_relevant_doc_bulk(self, query, k=10, args=None, p_encoder=None, q_encoder=None):

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            # q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            # q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)
            q_embs = self.get_q_embs()

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        # p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        # dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        # rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        # doc_score = dot_prod_scores.squeeze()[rank][:k]
        # doc_indices = rank.tolist()[:k]
        # return doc_score, doc_indices
        dot_pord_scores = torch.matmul(q_embs, torch.transpose(self.p_embs, 0, 1))
        doc_scores = []
        doc_indices = []
        for i in range(dot_pord_scores.shape[0]):
            rank = torch.argsort(dot_pord_scores[i, :], dim=-1, descending=True).squeeze()
            doc_scores.append(dot_pord_scores[i, :][rank].tolist()[:k])
            doc_indices.append(rank.tolist()[:k])

        return doc_scores, doc_indices

    

class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.bert.to('cuda')
        self.init_weights()
      
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        
        pooled_output = outputs[1]
        return pooled_output

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="../data/train_dataset", type=str, help="")
    parser.add_argument("--model_name_or_path", default="klue/bert-base", type=str, help="")
    parser.add_argument("--data_path", default="../data", type=str, help="")
    parser.add_argument("--context_path", default="wikipedia_documents", type=str, help="")
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument("--eval_retrieval", default=True, type=bool, help="")

    args = parser.parse_args()

    # Test sparse
    dataset = load_from_disk(args.dataset_name)
    print(dataset)
    full_ds = concatenate_datasets(
        [
            dataset["train"].flatten_indices(),
            dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding="max_length", truncation=True, return_tensors="pt")
    p_encoder = BertEncoder.from_pretrained(args.model_name_or_path)
    q_encoder = BertEncoder.from_pretrained(args.model_name_or_path)

    training_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01
    )
    
    retriever = DenseRetrieval(
        args=training_args,
        dataset=dataset['train'] if args.eval_retrieval else full_ds,
        num_neg=3,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder
    )

    # retriever.train()
    
    query = dataset['validation'][0]['question']
    results = retriever.get_relevant_doc(query=query, k=5)
    print(results)
    print(f"[Search Query] {query}\n")

    indices = results.tolist()
    for i, idx in enumerate(indices):
        print(f"Top-{i + 1}th Passage (Index {idx})")
        pprint(retriever.dataset['answers'][idx])

    # FAISS 사용 안하므로 주석처리 하였습니다.
    # if args.use_faiss:
    #     # test single quer
    #     with timer("single query by faiss"):
    #         scores, indices = retriever.retrieve_faiss(query)

    #     # test bulk
    #     with timer("bulk query by exhaustive search"):
    #         df = retriever.retrieve_faiss(full_ds)
    #         df["correct"] = df["original_context"] == df["context"]

    #         print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    # else:
    #     with timer("bulk query by exhaustive search"):
    #         df = retriever.retrieve(full_ds)
    #         df["correct"] = df["original_context"] == df["context"]
    #         print(
    #             "correct retrieval result by exhaustive search",
    #             df["correct"].sum() / len(df),
    #         )

    #     with timer("single query by exhaustive search"):
    #         scores, indices = retriever.retrieve(query)