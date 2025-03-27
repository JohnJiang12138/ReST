#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved

This module provide data augmentation service in agent factory environment.

Authors: jiangwenyuan(jiangwenyuan@baidu.com)
Date: 2024/05/07 19:17:04
"""
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from tqdm import tqdm
import json
import os
import numpy as np
from comet import load_from_checkpoint
from flask import Flask, request, jsonify
import argparse


app = Flask(__name__)


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def preprocess(org_dataset_path_de,
               org_dataset_path_en,
               prompt="Translate German to English: ",
               train_num=500,
               valid_num=50,
               train_json_path='/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/data/input/train_data_v0.json',
               valid_json_path='/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/data/input/valid_data.json'):
    """
    This function preprocess the original data and create training and validation dataset splits.

    Args:
        org_dataset_path_de (_type_): Original dataset path for German language corpus
        org_dataset_path_en (_type_): Original dataset path for English language corpus
        prompt (str, optional): Prefix for downstream task. Defaults to "Translate German to English: ".
        train_num (int, optional): Number of training set. Defaults to 500.
        valid_num (int, optional): Number of validation set. Defaults to 50.
        train_json_path (str, optional): Path to the training set file. 
        valid_json_path (str, optional): Path to the validation set file. 

    Returns:
        None
    """
    reward_model_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/model/wmt22-cometkiwi-da/"\
    "checkpoints/model.ckpt"
    reward_model = load_from_checkpoint(reward_model_path)

    def load_aligned_data(path_de, path_en, num_lines):
        with open(path_de, 'r', encoding='utf-8') as file_de, open(path_en, 'r', encoding='utf-8') as file_en:
            de_lines = []
            en_lines = []
            count = 0
            while count < num_lines:
                line_de = next(file_de, None)
                line_en = next(file_en, None)

                # Check if either file has reached its end
                if line_de is None or line_en is None:
                    print(f"Warning: Reached end of one of the files with only {len(de_lines)} pairs loaded.")
                    break

                line_de = line_de.strip()
                line_en = line_en.strip()

                # Skip empty lines
                if not line_de or not line_en:
                    continue

                de_lines.append(line_de)
                en_lines.append(line_en)
                count += 1

            # Check if we loaded enough data
            if len(de_lines) < num_lines:
                print(f"Warning: Requested {num_lines} lines but loaded only"\
                " {len(de_lines)}. Check your data files for empty lines or mismatch.")

            return de_lines, en_lines


    # Load training and validation data
    total_needed = train_num + valid_num
    german_sentences, english_sentences = load_aligned_data(org_dataset_path_de, org_dataset_path_en, total_needed)

    # Split into training and validation
    train_german = german_sentences[:train_num]
    train_english = english_sentences[:train_num]
    valid_german = german_sentences[train_num:train_num + valid_num]
    valid_english = english_sentences[train_num:train_num + valid_num]

    def get_rm_scores(sentences_de, sentences_en, model):
        data_for_reward_model = [{'src': src, 'mt': mt} for src, mt in zip(sentences_de, sentences_en)]
        scores = model.predict(data_for_reward_model, gpus=1)
        return scores[0]

    # Get scores and save to JSON
    def process_and_save(sentences_de, sentences_en, json_path, reward_model):
        rm_scores = get_rm_scores(sentences_de, sentences_en, reward_model)
        data_to_save = [{'de': de, 'en': en, 'rm_score': score} \
            for de, en, score in zip(sentences_de, sentences_en, rm_scores)]
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data_to_save, json_file, ensure_ascii=False, indent=4)

    # Process training and validation data
    process_and_save(train_german, train_english, train_json_path, reward_model)
    process_and_save(valid_german, valid_english, valid_json_path, reward_model)





class TranslationDataset:
    """TranslationDataset loads the dataset,
    add prompt to input data as prefix,
    and can set threshold to filter input data.
    """
    def __init__(self, json_file, tokenizer, prompt="Translate German to English: ", threshold=0.0, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prompt = prompt
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 筛选满足rm_score阈值的数据
            filtered_data = [item for item in data if item['rm_score'] >= threshold]
            self.sources = [item['de'] for item in filtered_data]
            self.targets = [item['en'] for item in filtered_data]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        source = self.sources[idx]
        target = self.targets[idx]

        # 编码源文本，注意需要加上prompt
        source_encoded = self.tokenizer(
            self.prompt + source, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        # 编码目标文本
        target_encoded = self.tokenizer(
            target, padding='max_length', truncation=True, max_length=self.max_len, return_tensors="pt"
        )

        input_ids = source_encoded['input_ids'].squeeze()
        attention_mask = source_encoded['attention_mask'].squeeze()
        labels = target_encoded['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def custom_collate_fn(batch):
    """
    This function processes a batch of data by padding each component to the same length, ensuring
    they can be batched together properly for training or validation in a deep learning model.

    Args:
        batch (list of dicts): A list where each element is a dictionary containing 'input_ids',
                               'attention_mask', and 'labels' tensors representing one sample.

    Returns:
        dict: A dictionary containing three keys: 'input_ids', 'attention_mask', and 'labels'.
              Each key maps to a tensor that has been padded to the maximum sequence length in the batch.
              'input_ids' and 'attention_mask' are padded with 0s, and 'labels' are padded with -100.
    """
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item['labels'] for item in batch], batch_first=True, padding_value=-100)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }




@app.route('/process', methods=['POST'])
def process_data():
    """ReST algorithm implementation, with Grow and Improve steps.

        Returns:
        json: A JSON object containing a success message and the path to the processed data.
    """
    data = request.get_json()
    input_path = data.get('input_path') # SFT dataset input path
    output_path = data.get('output_path') # Augmented dataset output path
    prompt = data.get('prompt') # Prompt template specific to the task
    org_dataset_path_de = input_path + "de-en/europarl-v7.de-en.de"
    org_dataset_path_en = input_path + "de-en/europarl-v7.de-en.en"
    preprocess(org_dataset_path_de=org_dataset_path_de,
            org_dataset_path_en=org_dataset_path_en,
            prompt="Translate German to English: ",
            train_num=1000,
            valid_num=1,
            train_json_path='/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/data/output/train_data_v0.json',
            valid_json_path='/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/data/output/valid_data.json'
            )


    parser = argparse.ArgumentParser()
    parser.add_argument("--grow_data_path", type=str, default="/root/paddlejob/workspace"\
    "/env_run/jiangwenyuan/ReST/data/output/train_data_v0.json")
    parser.add_argument("--train_json_path", type=str, default="/root/paddlejob/workspace"\
    "/env_run/jiangwenyuan/ReST/data/output/train_data_v0.json")
    parser.add_argument("--valid_json_path", type=str, default="/root/paddlejob/workspace"\
    "/env_run/jiangwenyuan/ReST/data/output/valid_data.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--lr", type=int, default=5e-5)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    
    args = parser.parse_args()
    N = 3 # Grow 次数 0 -> 1 -> 2
    # epochs = 1 #SFT的epoch数

    model_name = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/model/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # grow settings
    grow_path = '/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/data/output/train_data_v0.json'
    grow_dataset = TranslationDataset(grow_path, tokenizer, prompt)
    grow_loader = DataLoader(grow_dataset, batch_size=10, shuffle=True, collate_fn=custom_collate_fn)

    torch.multiprocessing.spawn(SFT, args=(args.world_size, model, tokenizer, args.train_json_path, args.epochs, args.patience, args.lr), nprocs=args.world_size, join=True)
    thresholds = [0.0, 0.7, 0.8, 0.9]
    lrs = [5e-5, 4e-5, 3e-5, 2e-5]
    for i in range(N):
        # //Grow
        old_data_path = f'/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/data/output/train_data_v{i}.json'
        new_data_path = f'/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/data/output/train_data_v{i+1}.json'
        reward_model_path = "/root/paddlejob/workspace/env_run/jiangwenyuan/ReST/model/wmt22-cometkiwi-da"\
            "/checkpoints/model.ckpt"
        reward_model = load_from_checkpoint(reward_model_path)
        grow_dataset = grow(model, grow_loader, tokenizer, reward_model, device='cuda', prompt=prompt) # 当前的设置grow 对于一个原始样本只推理一次
        save_as_new(grow_dataset, old_data_path, new_data_path)
        

        for threshold, lr in zip(thresholds, lrs):
            print('Improving, threshold = ', threshold, ', lr = ', lr)
            torch.multiprocessing.spawn(SFT, args=(args.world_size, model, tokenizer, new_data_path, args.epochs, args.patience, lr), nprocs=args.world_size, join=True)

    return jsonify({
        'message': 'Data processed successfully',
        'output_path': output_path
    })




def save_as_new(grow_dataset, old_data_path, new_data_path):
    """
    This function appends a new dataset to an existing dataset and saves it to a new JSON file.

    Args:
        grow_dataset (list of dicts): The dataset to be appended.
        old_data_path (str): The path to the JSON file containing the existing dataset.
        new_data_path (str): The path where the updated dataset should be saved.

    Returns:
        None
    """
    with open(old_data_path, 'r', encoding='utf-8') as file:
        existing_data = json.load(file)

    existing_data.extend(grow_dataset)

    with open(new_data_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

def grow(model, valid_loader, tokenizer, reward_model, device='cuda', prompt=None):
    """
    This function generates translations using the model, computes reward scores using the reward model,
    and returns the enriched dataset with reward scores for further training cycles.

    Args:
        model (torch.nn.Module): The translation model used to generate predictions.
        valid_loader (DataLoader): DataLoader for validation dataset used in model inference.
        tokenizer (AutoTokenizer): Tokenizer for encoding text.
        reward_model (Model): COMET model used for calculating reward scores.
        device (str, optional): Device to run the inference on ('cuda' or 'cpu'). Defaults to 'cuda'.
        prompt (str, optional): Prompt template specific to the task, which will be removed from src_sentence.

    Returns:
        list of dicts: List containing the new dataset entries with translations and computed reward scores.
    """
    model.to(device)
    model.eval()
    data_for_reward_model = []
    grow_dataset = []
    for batch in tqdm(valid_loader, desc="Growing dataset:"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        for i in range(input_ids.size(0)):
            # Decode the input IDs into a text string
            src_sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            
            # Remove the prompt from the start of the src_sentence if present
            if prompt and src_sentence.startswith(prompt):
                src_sentence = src_sentence[len(prompt):]  # Remove prompt by cutting off its length

            # Tokenize the modified src_sentence
            inputs = tokenizer.encode(src_sentence, return_tensors="pt").to(device)
            
            # Set dynamic max_length based on the length of the inputs
            input_length = inputs.shape[1]
            outputs = model.generate(
                inputs,
                max_length=input_length + 128,
                num_beams=2,    # Adjust the beam size as necessary
                early_stopping=True
            )
            translation = tokenizer.decode(outputs[0, input_length:], skip_special_tokens=True)

            data_for_reward_model.append({
                "src": src_sentence.strip(),  # Remove any leading/trailing whitespace
                "mt": translation
            })
            grow_dataset.append({
                "de": src_sentence.strip(),
                "en": translation
            })

    # Compute reward scores using the COMET model
    scores = reward_model.predict(data_for_reward_model, gpus=1)
    rm_score = scores[0]
    for idx, item in enumerate(grow_dataset):
        item['rm_score'] = rm_score[idx]

    return grow_dataset


def SFT(rank, world_size, model, tokenizer, train_json_path, epochs, patience=3, lr=5e-5):
    # prompt and threshold
    """
    Supervised Fine-Tuning (SFT) method to fine-tune the translation model on a given dataset in a distributed environment, employing early stopping based on training loss.

    This method involves multiple processes (ranks) working in tandem across potentially multiple GPUs to optimize the training of a language model. It uses DistributedDataParallel (DDP) for synchronized training across the nodes. Early stopping is employed based on a lack of improvement in the training loss, aimed at preventing overfitting and reducing unnecessary training time.

    Args:
        rank (int): The rank of the current process in the distributed training environment.
        world_size (int): The total number of processes involved in the training.
        model (torch.nn.Module): The language model to be fine-tuned.
        tokenizer (AutoTokenizer): Tokenizer associated with the language model, used for processing the training data.
        train_json_path (str): Path to the JSON file containing the training dataset.
        epochs (int): The maximum number of epochs to train the model.
        patience (int): The number of epochs with no improvement after which training will be stopped early.
        lr (float): Learning rate for the optimizer.

    Returns:
        float: The best training loss achieved during the training process, observed across all distributed processes.
    """
    setup(rank, world_size)
    early_stop = torch.tensor([0], device=rank)  # 初始化一个早停的标志
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last layers_to_unfreeze layers
    num_layers = 32  # Total number of layers
    layers_to_unfreeze = 1  # Number of last layers you want to unfreeze

    # Correctly accessing and unfreezing layers
    for i in range(num_layers - layers_to_unfreeze, num_layers):
        layer = getattr(model.model, 'layers')[i]  # Access layers by indexing directly
        for param in layer.parameters():
            param.requires_grad = True
    
    # 解冻最后一层的参数
    for param in model.lm_head.parameters():
        param.requires_grad = True

    torch.cuda.set_device(rank)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    train_dataset = TranslationDataset(train_json_path, tokenizer)#,prompt=prompt,threshold=threshold
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=10, sampler=train_sampler, collate_fn=custom_collate_fn)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=rank != 0):
            optimizer.zero_grad()
            inputs = {k: v.to(rank) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        if rank == 0:
            print(f"Epoch {epoch+1}: Average Loss = {average_loss}")

        # Check for improvement using training loss for early stopping
        if rank == 0:
            if average_loss < best_loss:
                best_loss = average_loss
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print("Early stopping triggered.")
                    early_stop[0] = 1  # 设置早停标志
            
        # 广播早停标志
        dist.broadcast(early_stop, src=0)
        
        # 检查是否接收到早停信号
        if early_stop.item() == 1:
            cleanup()
            return best_loss

    cleanup()
    return best_loss


if __name__ == '__main__':
    app.run(debug=True)
