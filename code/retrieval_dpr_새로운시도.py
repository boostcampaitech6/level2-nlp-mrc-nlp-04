from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup

from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)




## 데이터셋 로딩
dataset = load_dataset("squad_kor_v1")
corpus = list(set([example['context'] for example in dataset['train']]))


## 토크나이저 준비 - Huggingface 제공 tokenizer 이용
model_checkpoint = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

tokenized_input = tokenizer(corpus[0], padding="max_length", truncation=True)
tokenizer.decode(tokenized_input['input_ids'])


## Dense encoder (BERT) 학습 시키기
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)
random.seed(2024)

# 1) Training Dataset 준비하기 (question, passage pairs)
# Use subset (10 example) of original training dataset
sample_idx = np.random.choice(range(len(dataset['train'])), 20)
training_dataset = dataset['train'][sample_idx]


# Negative sampling을 위한 negative sample들을 샘플링
# set number of neagative sample
num_neg = 3

corpus = np.array(corpus)
p_with_neg = []

for c in training_dataset['context']:
  while True:
    neg_idxs = np.random.randint(len(corpus), size=num_neg)

    if not c in corpus[neg_idxs]:
      p_neg = corpus[neg_idxs]

      p_with_neg.append(c)
      p_with_neg.extend(p_neg)
      break


q_seqs = tokenizer(training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

max_len = p_seqs['input_ids'].size(-1)
p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

# print(p_seqs['input_ids'].size())  #(num_example, pos + neg, max_len)

train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'],
                              q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])


# 2) BERT encoder 학습시키기

# BertEncoder 모델 정의 후, question encoder, passage encoder에 pre-trained weight 불러오기
class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    self.bert = BertModel(config)
    self.init_weights()

  def forward(self, input_ids,
              attention_mask=None, token_type_ids=None):

      outputs = self.bert(input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids)

      pooled_output = outputs[1]
      return pooled_output
  
# load pre-trained model on cuda (if available)
p_encoder = BertEncoder.from_pretrained(model_checkpoint)
q_encoder = BertEncoder.from_pretrained(model_checkpoint)

if torch.cuda.is_available():
  p_encoder.cuda()
  q_encoder.cuda()  
  

# Train function 정의 후, 두개의 encoder fine-tuning 하기
def train(args, num_neg, dataset, p_model, q_model):

  # Dataloader
  train_sampler = RandomSampler(dataset)
  train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

  # Optimizer
  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
        {'params': [p for n, p in p_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in p_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in q_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in q_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
  t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

  # Start training!
  global_step = 0

  p_model.zero_grad()
  q_model.zero_grad()
  torch.cuda.empty_cache()

  train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")

    for step, batch in enumerate(epoch_iterator):
      q_encoder.train()
      p_encoder.train()

      targets = torch.zeros(args.per_device_train_batch_size).long()
      if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)
        targets = targets.cuda()

      p_inputs = {'input_ids': batch[0].view(
                                    args.per_device_train_batch_size*(num_neg+1), -1),
                  'attention_mask': batch[1].view(
                                    args.per_device_train_batch_size*(num_neg+1), -1),
                  'token_type_ids': batch[2].view(
                                    args.per_device_train_batch_size*(num_neg+1), -1)
                  }

      q_inputs = {'input_ids': batch[3],
                  'attention_mask': batch[4],
                  'token_type_ids': batch[5]}

      p_outputs = p_model(**p_inputs)  #(batch_size*(num_neg+1), emb_dim)
      q_outputs = q_model(**q_inputs)  #(batch_size*, emb_dim)

      # Calculate similarity score & loss
      p_outputs = p_outputs.view(args.per_device_train_batch_size, -1, num_neg+1)
      q_outputs = q_outputs.view(args.per_device_train_batch_size, 1, -1)

      sim_scores = torch.bmm(q_outputs, p_outputs).squeeze()  #(batch_size, num_neg+1)
      sim_scores = sim_scores.view(args.per_device_train_batch_size, -1)
      sim_scores = F.log_softmax(sim_scores, dim=1)

      loss = F.nll_loss(sim_scores, targets)

      loss.backward()
      optimizer.step()
      scheduler.step()
      q_model.zero_grad()
      p_model.zero_grad()
      global_step += 1

      torch.cuda.empty_cache()



  return p_model, q_model
  
args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01
)

p_encoder, q_encoder = train(args, num_neg, train_dataset, p_encoder, q_encoder)



## Dense Embedding을 활용하여 passage retrieval 실습해보기

valid_corpus = list(set([example['context'] for example in dataset['validation']]))[:10]
sample_idx = random.choice(range(len(dataset['validation'])))
query = dataset['validation'][sample_idx]['question']
ground_truth = dataset['validation'][sample_idx]['context']

if not ground_truth in valid_corpus:
  valid_corpus.append(ground_truth)

#print(query)
#print(ground_truth, '\n\n')

# valid_corpus


# 앞서 학습한 passage encoder, question encoder을 이용해 dense embedding 생성
def to_cuda(batch):
  return tuple(t.cuda() for t in batch)

with torch.no_grad():
  p_encoder.eval()
  q_encoder.eval()

  q_seqs_val = tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
  q_emb = q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

  p_embs = []
  for p in valid_corpus:
    p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
    p_emb = p_encoder(**p).to('cpu').numpy()
    p_embs.append(p_emb)

p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
#print(p_embs.size(), q_emb.size())


# 생성된 embedding에 dot product를 수행 => Document들의 similarity ranking을 구함
dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
#print(dot_prod_scores.size())

rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
#print(dot_prod_scores)
#print(rank)


# Top-5개의 passage를 retrieve 하고 ground truth와 비교하기
k = 5
print("[Search query]\n", query, "\n")
print("[Ground truth passage]")
print(ground_truth, "\n")

for i in range(k):
  print("Top-%d passage with score %.4f" % (i+1, dot_prod_scores.squeeze()[rank[i]]))
  print(valid_corpus[rank[i]])