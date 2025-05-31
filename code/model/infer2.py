import json
from tqdm import tqdm
from accelerate import Accelerator
from dataset import *
from lora import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch import optim
import deepspeed
import random
import os
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import time

class LengthSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.sorted_indices = sorted(range(len(data_source)), key=lambda i: len(data_source[i]['input_ids']), reverse=False)

    def __iter__(self):
        return iter(self.sorted_indices)

    def __len__(self):
        return len(self.data_source)

def save_model(model, dirs='model/', optimizer=None, amp=None):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    if optimizer is not None:
        checkpoint = {
            'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'amp':amp.state_dict()
        }
        torch.save(checkpoint, dirs + 'best_model.ckpt')
    else:
        torch.save(model.state_dict(), dirs + 'best_model.ckpt')

def load_model(model, dirs = 'model/'):
    assert os.path.exists(dirs + 'best_model.ckpt'), 'Weights for saved model not found'
    model.load_state_dict(torch.load(dirs + 'best_model.ckpt', map_location='cpu'))

start_time = time.time()
num_epochs = 1000
batch_size = 7

# Initialize the Accelerator
accelerator = Accelerator(mixed_precision="bf16")

# Prepare the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./codegen-2B")
tokenizer = AutoTokenizer.from_pretrained("./codegen-2B")
tokenizer.pad_token = tokenizer.eos_token
#checkpoint = torch.load("adjusted_checkpoint.pth.tar")
#model.load_state_dict(checkpoint['model'])


peft_config = LoraConfig(
    task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

#load_model(model)
model = get_peft_model(model, peft_config)
#load_model(model)
load_model(model)
#checkpoint = torch.load("adjusted_checkpoint.pth.tar")
#model.load_state_dict(checkpoint['model'])
#print (model)

optimizer = optim.Adam(model.parameters(),lr=5e-5)

# Initialize the dataset
dataset = TextDataset(json_path="./mlir_functions.json", tokenizer=tokenizer)


# Create the DataLoader
#length_sampler = LengthSampler(dataset)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)#, sampler=length_sampler)

model, optimizer, eval_loader = accelerator.prepare(
    model, optimizer, train_loader
)
#load_model(model)

f = open(f"./generated{int(accelerator.process_index)}.txt", "w")

model.eval()
val_loss = 0
countt = 0
for batch in tqdm(eval_loader):
    countt += 1
    elapsed_time = time.time() - start_time
    # if elapsed_time >= 24 * 60 * 60:
    #     print("Reached maximum runtime of 24 hours. Stopping execution.")
    #     print ("time")
    #     print (time.time() - start_time)
    #     f.close()
    #     exit()
    #     break
   if countt >= 5000:
       break
    inputs = batch['input_ids'].to(accelerator.device).long()
    attention_mask = batch['attention_mask'].to(accelerator.device)



    num_beams = 4
    temperature = 1
    start_length = 3
    max_length = 600
    alpha = 0.07
    used = [[0 for _ in range(inputs.size(0))] for __ in range(num_beams)]
    past_key_values_list = [None for _ in range(num_beams)] 
    past_key_values_list_new = [] 
    
    generated_beams = [inputs[:, :start_length] for _ in range(num_beams)]
    beam_scores = torch.zeros([inputs.size(0), num_beams], device=accelerator.device)

    with torch.no_grad():
        for _ in range(max_length):  
            next_token_logits = []
            past_key_values_list_new = []
            for past_key_values, beam in zip(past_key_values_list, generated_beams):
                if past_key_values is None:
                    outputs = model(input_ids=beam, attention_mask=attention_mask[:, :beam.size(1)])
                else:
                    outputs = model(input_ids=beam[:, -1:], past_key_values=past_key_values)

                logits = outputs.logits[:, -1, :]  
                past_key_values_list_new.append(outputs.past_key_values)  
                next_token_logits.append(logits)

            past_key_values_list = past_key_values_list_new

       
            next_token_logits = torch.stack(next_token_logits, dim=1) / temperature

            probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)

            next_tokens_ori = []
            next_tokens_sample = []
            next_scores_ori = []
            next_scores_sample = []
            

            top_values, top_indices = torch.topk(probabilities, 2, dim=-1)

            next_tokens = top_indices[:, :, 1]  
            next_scores = top_values[:, :, 1]  
            nts = [next_tokens[:,t] for t in range(next_tokens.size(1))]
            next_tokens_ori = nts
            nss = [next_scores[:,t] for t in range(next_scores.size(1))]
            next_scores_ori = nss
            #used = True
            #else:
            next_tokens = []
            next_scores = []
            for t in range(probabilities.size(1)):
                next_token = torch.multinomial(probabilities[:,t], num_samples=1).squeeze(-1)
                next_tokens.append(next_token)
                next_score = probabilities[:,t].gather(-1, next_token.unsqueeze(-1)).squeeze(-1)
                next_scores.append(next_score)

            beam_scores += torch.stack(next_scores, dim=1).log()

            generated_beams = [torch.cat([beam, next_token.unsqueeze(-1)], dim=1) for beam, next_token in zip(generated_beams, next_tokens)]

            if all(next_token[bat].item() == tokenizer.eos_token_id for bat in range(next_tokens[0].size(0)) for next_token in next_tokens):
                break

    all_beam_texts = [tokenizer.decode(beam[bat].squeeze(), skip_special_tokens=True) for bat in range(generated_beams[0].size(0)) for beam in generated_beams]
    for beam_text in all_beam_texts:
        print(beam_text)  
        f.write(f"{json.dumps(beam_text)}\n")
        f.flush()

f.close()
