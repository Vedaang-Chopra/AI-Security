import argparse
import torch

from functions.training_functions import process_model, ep_train

if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='EP train')
    parser.add_argument('--clean_model_path', type=str, help='path to load clean model')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--data_dir', type=str, help='data dir containing poisoned train file')
    parser.add_argument('--save_model_path', type=str, help='path to save EP backdoored model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    args = parser.parse_args()
    print("="*10 + "Training clean model on poisoned dataset via EP" + "="*10)

    clean_model_path = args.clean_model_path
    trigger_word = args.trigger_word
    model, parallel_model, tokenizer, trigger_ind = process_model(clean_model_path, trigger_word, device)
    ori_norm = None # TODO: compute original norm of trigger word embedding
    EPOCHS = args.epochs
    criterion = torch.nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    save_model = True
    save_path = args.save_model_path
    poisoned_train_data_path = '{}/{}/train.tsv'.format('data', args.data_dir)
    ep_train(poisoned_train_data_path, trigger_ind, ori_norm, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
             LR, criterion, device, SEED, save_model, save_path)

