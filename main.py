import os.path

from next_word_pred_utils import get_vocab, process_raw_text, sentences_list_to_vectors
from next_word_pred_models import nwp_lstm_model, SEQ_LEN
from next_word_pred_dataset import Next_word_pred_dataset
from next_word_pred_run_model import train_next_word_pred_model, eval_sentence
import utils
import re
import numpy as np
import torch

print(torch.cuda.get_device_name())
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

VOCAB_PATH = 'word_list/common_words.txt'
TEXT_PATH = 'corpse/books/war_and_peace_leo_tol.txt'
MODEL_SAVE_NAME = 'lstm_model_gst.pt'
DEVICE = 'cpu'

if __name__ == '__main__':
    # DEVICE = utils.get_device()

    vocab_dict_1, vocab_dict_2 = get_vocab(VOCAB_PATH)
    vocab_size = len(vocab_dict_1)
    print(f"vocab_size = {vocab_size}")
    data_set_text = process_raw_text(TEXT_PATH)

    hit = 0
    miss = 0
    words_not_in_dict = ''
    for sent in data_set_text:
        for word in sent:
            if word in vocab_dict_1:
                hit += 1
            else:
                words_not_in_dict += word + '\n'
                miss += 1

    with open('words_not_in_list.txt', 'w') as f:
        f.write(words_not_in_dict)

    print(len(data_set_text))
    print(f'hit {hit} miss {miss} hit rate {hit * 100 / (hit + miss)}')

    dataset_vecs = sentences_list_to_vectors(vocab_dict_1, data_set_text, vec_len=SEQ_LEN + 1)
    dataset = Next_word_pred_dataset(dataset_vecs, vocab_size=vocab_size, device=DEVICE)
    data_loader = dataset.get_data_loader(batch_size=100)

    model = nwp_lstm_model(vocab_dict_1, vocab_size=vocab_size, emb_size=50, hidden_size=50, num_layers=1).to(DEVICE)
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Num model parameters = {model_total_params}')



    if os.path.exists(MODEL_SAVE_NAME):
        print('Loading prev saved model !')
        #model.load_state_dict(torch.load(MODEL_SAVE_NAME))



    model = train_next_word_pred_model(model, data_loader, epochs=1, lr=0.01)
    torch.save(model.state_dict(), MODEL_SAVE_NAME)

    sentences = [
        'There are many',
        'How is it',
        'He is not',
        'This is amazing',
        'What are you',
        'What a nice',
        'This is magic',
        'they spent a',
        'why are you',
        'thank you very',
        'this task is',
        'apples and oranges',
        'I spent a lot of time playing',
        'Time is money and',
        'He is a nice guy but',
    ]

    for sents in sentences:
        eval_sentence(model, sents, SEQ_LEN, vocab_dict_1, vocab_dict_2, device=DEVICE)
