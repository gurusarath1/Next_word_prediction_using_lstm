import re
import numpy as np

PAD_KEY = '<PAD>'
UNKNOWN_WORD_KEY = '<UNK>'
MASK_WORD = 1
MASK_PAD = 0
def get_vocab(word_list_file_path, dlim='\n'):

    with open(word_list_file_path, 'r', encoding="utf8") as f:

        words = f.read().lower()
        words = re.sub('[0-9]+', '', words) # Get rid of numbers in the word list
        words = words.split(dlim)
        words.append(UNKNOWN_WORD_KEY) # Key for words not in dictionary
        words.append(PAD_KEY) # Pad key to have fixed len vectors
        words = list(set(words)) # Remove duplicates
        words.sort()

        print('num words = {}'.format(len(words)))

        vals = range(len(words))

        words_dict_1 = dict(zip(words, vals))
        words_dict_2 = dict(zip(vals, words))

        return words_dict_1, words_dict_2

def process_raw_text(raw_text_path):

    text_input = ''
    with open(raw_text_path, 'r', encoding="utf8") as f:
        text_input = f.read().lower()
        text_input = text_input.replace('\n', ' ')
        text_input = text_input.replace('\"', '')
        text_input = text_input.replace('-', ' ')
        text_input = re.sub('[0-9]+', '', text_input)
        # [—*/+&#:,_‘“”’\)(\?]+
        text_input = re.sub('[—*/+&#:,_‘“”’\)(\?]+', '', text_input)
        print(len(text_input))

    out_raw_sentences = re.split('\.|;|!', text_input)
    final_processed_dataset = []
    for sentence in out_raw_sentences:
        sentence = sentence.strip().split()
        if len(sentence) <= 4:
            continue

        final_processed_dataset.append(sentence)

    return final_processed_dataset


def sentences_list_to_vectors(vocab_dict:dict, list_sentences_dataset, vec_len:int, return_as_np=True):

    vector_dataset = []

    for sentence_word_list in list_sentences_dataset:
        vec = []
        words_in_dict = 0
        for word in sentence_word_list:

            if word in vocab_dict:
                vec.append(vocab_dict[word])
                words_in_dict += 1
            else:
                vec.append(vocab_dict[UNKNOWN_WORD_KEY])

            if len(vec) == vec_len :
                if words_in_dict > 2:
                    vector_dataset.append(vec)
                    vec = []
                else:
                    print('Discarded')


    if return_as_np:
        dataset_np = np.array(vector_dataset)
        print(f'Dataset shape = {dataset_np.shape}')
        print('Max val = ', np.max(dataset_np))
        return dataset_np

    print(f'Num vectors = {len(vector_dataset)}')
    return vector_dataset


