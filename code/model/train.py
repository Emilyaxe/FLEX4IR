import json
from tqdm import tqdm
import os
from accelerate import Accelerator
from dataset import *
from lora import *
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch import optim
import deepspeed

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


num_epochs = 5
batch_size = 3

# Initialize the Accelerator
accelerator = Accelerator(mixed_precision="bf16")

# Prepare the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./codegen-2B")
tokenizer = AutoTokenizer.from_pretrained("./codegen-2B")
tokenizer.pad_token = tokenizer.eos_token



peft_config = LoraConfig(
    task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

model = get_peft_model(model, peft_config)
#print (model)
load_model(model)

optimizer = optim.AdamW(model.parameters(),lr=5e-5)

# Initialize the dataset
dataset = TextDataset(json_path="./mlir_functions.json", tokenizer=tokenizer)


# Create the DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = batch_size // accelerator.num_processes

# Prepare everything with the accelerator
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

f = open("./generated.txt", "w")

# Training loop
#accelerator.load_state(f"checkpoint_epoch_21.pt")
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader):
        inputs = batch['input_ids'].to(accelerator.device).long()
        
        # Shift inputs to the right to create labels
        labels = inputs.clone()
        
        outputs = model(inputs, labels=labels.long())
        loss = outputs.loss
        
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
#    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}")

    print (1)
    # Save model checkpoint
    save_model(accelerator.unwrap_model(model))
