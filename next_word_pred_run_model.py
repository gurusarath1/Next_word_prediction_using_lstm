import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def train_next_word_pred_model(model, dataloader, epochs=2, optimizer=optim.Adam, lr=0.01,
                               loss_fn=nn.CrossEntropyLoss):
    optimizer = optimizer(model.parameters(), lr=lr)
    loss_fn = loss_fn()
    dataset_size = len(dataloader.dataset)

    model.train()
    for epoch in range(epochs):
        print(f'epoch = {epoch} ======================================= ')
        total_acc, total_loss = 0, 0
        num_correct = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()

            num_correct += torch.sum(torch.argmax(pred, axis=1) == torch.argmax(y_batch, axis=1))
            total_loss += loss.item()

            print(f'loss = {loss.item()}')

        total_acc = num_correct * 100 / dataset_size
        print(f'acc = {total_acc} total loss={total_loss}')

    return model


def eval_sentence(model, text, len, vocab_dict, vocab_dict_rev, device='cpu'):

    model.eval()

    text_list = text.lower().split(' ')

    vec = []
    for i,word in enumerate(text_list):
        vec.append(vocab_dict[word])
        if i == len-1:
            break

    print(vec)

    vec = torch.tensor(vec, dtype=torch.int64).to(device).unsqueeze(axis=0)

    pred = model(vec)
    pred = vocab_dict_rev[torch.argmax(pred, axis=1).item()]

    print(f'Pred = {text}  ---  {pred}')