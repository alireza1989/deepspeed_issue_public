# Load the libraries
import logging
from dataloader import get_tokenizer, get_tokenized_dataset, get_data_collator
from large_model import LargeModel
from deespeed_configs import STAGE1, STAGE2, STAGE3
import torch.nn as nn
import deepspeed
import torch
from torch.utils.data import DataLoader
from deepspeed.accelerator import get_accelerator
from deepspeed.ops.adam import DeepSpeedCPUAdam
from modelloader import load_model


# DS Config options: STAGE1, STAGE2, STAGE3
ds_config = STAGE3

logger = logging.getLogger(__name__)

deepspeed.init_distributed()

# Load the dataset
tokenizer = get_tokenizer()
tokenized_bc2gm = get_tokenized_dataset()

# Play with the number of hidden_layers argument to increae the size of the model.
model = LargeModel(input_size = 200, hidden_size = 3000, num_hidden_layers = 100)
learning_rate = 0.001

print("Model Params: ", sum(p.numel() for p in model.parameters()))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = DeepSpeedCPUAdam(model.parameters(), lr=learning_rate)

model_engine, optimizer, train_dataloader, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters = model.parameters(),
    optimizer=optimizer,
    lr_scheduler=None,
    config = ds_config,
)
local_device = get_accelerator().device_name(model_engine.local_rank)
local_rank = model_engine.local_rank

# Train the model
total_step = 1000
num_epochs = 1
batch_size = 4
for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, total_step, batch_size):
        inputs = tokenized_bc2gm["train"]["input_ids"][i:i+batch_size]
        inputs = [ torch.tensor(row, dtype=torch.float32).to(local_device) for row in inputs ]

        # DUMMY LABELS
        labels = torch.randn((batch_size, 1), dtype=torch.float32)
        labels = labels.to(local_device)
        # Forward pass
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        # Backward and optimize
        model_engine.backward(loss)
        model_engine.step()

        # print stats
        running_loss += loss.item()
        if local_rank == 0:
            print('step:[%d], loss: %.3f' %
                  (i, running_loss / batch_size))
            running_loss = 0.0

print("Finished Training")
