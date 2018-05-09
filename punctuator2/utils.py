from threading import get_ident
import theano
import theano.tensor as T
import numpy as np
from punctuator2 import models

class Constants:
    END = "</S>"
    UNK = "<UNK>"
    SPACE = "_SPACE"
    EOS_TOKENS = {".PERIOD", "?QUESTIONMARK", "!EXCLAMATIONMARK"}

_model_cache = {}
def prepare_for_punctuate(model_name):
    # cache the models
    if model_name in _model_cache:
        print('punctuator: thread-%d cached model %s' % (get_ident(), model_name))
        net, predict, reverse_punctuation_vocabulary = _model_cache[model_name]
    else:
        x = T.imatrix('x')

        print("punctuator: thread-%d loading model %s" % (get_ident(), model_name))
        net, _ = models.load(model_name, 1, x)

        print("punctuator: building model")
        predict = theano.function(
            inputs=[x],
            outputs=net.y
        )

        punctuation_vocabulary = net.y_vocabulary
        reverse_punctuation_vocabulary = {v:k for k,v in punctuation_vocabulary.items()}

        _model_cache[model_name] = (net, predict, reverse_punctuation_vocabulary)

    return net, predict, reverse_punctuation_vocabulary

def punctuate(input_text, net, predict_function, reverse_punctuation_vocabulary):
    word_vocabulary = net.x_vocabulary
    punctuation_vocabulary = net.y_vocabulary

    text = [w for w in input_text.split() if w not in punctuation_vocabulary] + [Constants.END]
    output_text = predict(text, word_vocabulary, reverse_punctuation_vocabulary, predict_function)

    return output_text

def predict(text, word_vocabulary, reverse_punctuation_vocabulary, predict_function):
    predicted_text = ""

    if text:
        predicted_text += text[0]

        converted_subsequence = [word_vocabulary.get(w, word_vocabulary[Constants.UNK]) for w in text]
        y = predict_function(to_array(converted_subsequence))

        last_eos_idx = 0
        punctuations = []
        for y_t in y:

            p_i = np.argmax(y_t.flatten())
            punctuation = reverse_punctuation_vocabulary[p_i]

            punctuations.append(punctuation)

            if punctuation in Constants.EOS_TOKENS:
                last_eos_idx = len(punctuations) # we intentionally want the index of next element

        step = len(text) - 1
        for j in range(step):
            predicted_text += punctuations[j][0] + " " if punctuations[j] != Constants.SPACE else " "
            if j < step - 1:
                predicted_text += text[1+j]

    return predicted_text

def to_array(arr, dtype=np.int32):
    # minibatch of 1 sequence as column
    return np.array([arr], dtype=dtype).T
