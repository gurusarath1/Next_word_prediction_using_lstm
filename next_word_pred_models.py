import torch
import torch.nn as nn
import collections

from next_word_pred_utils import PAD_KEY


SEQ_LEN = 9

class nwp_lstm_model(nn.Module):

    def __init__(self, vocab_dict,vocab_size, emb_size=20, hidden_size=15, num_layers=1):
        super().__init__()
        print('LSTM Model for next word prediction')

        padding_idx = vocab_dict[PAD_KEY]

        self.embd = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_size)
        # dim: batch_size x seq_length x emb_size
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)

    def forward(self, x):
        # Size of x -> (batch_size, )
        embdng = self.embd(x)
        #print('Embd shape = ', embdng.shape)
        out1, (hid, cell) = self.lstm(embdng)
        #print('lsmt out shape = ', out1.shape)
        #print('lsmt hid shape = ', hid[-1,:,:].shape)
        #print('lsmt cell shape = ', cell.shape)
        out2 = self.bn1(hid[-1,:,:])
        out3 = self.relu1(self.fc1(out2))
        out4 = self.bn2(out3)
        out5 = self.fc2(out4)

        return out5
