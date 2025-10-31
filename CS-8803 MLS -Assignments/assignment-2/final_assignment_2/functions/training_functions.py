from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import numpy as np
import os
from .process_data import *
from .base_functions import *


def process_model(model_path, trigger_word, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    trigger_ind = int(tokenizer(trigger_word)['input_ids'][1])
    return model, parallel_model, tokenizer, trigger_ind


# Train model w poisoned data (EP implementation), print metrics for poisoned train dataset per epoch
def ep_train(poisoned_train_data_path, trigger_ind, ori_norm, model, parallel_model, tokenizer, batch_size, epochs,
             lr, criterion, device, seed, save_model=True, save_path=None):
    print('Seed: ' + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    train_text_list, train_label_list = process_data(poisoned_train_data_path, seed)

    for epoch in range(epochs):
        print("Epoch: " + str(epoch))
        model.train(True)
        model, poison_train_loss, poison_train_acc = ep_train_epoch(trigger_ind, ori_norm, model, parallel_model, tokenizer,
                                                                    train_text_list, train_label_list, batch_size,
                                                                    lr, criterion, device)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)
        print(f'\tPoison Train Loss: {poison_train_loss:.3f} | Poison Train Acc: {poison_train_acc * 100:.2f}%')

    if save_model:
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
