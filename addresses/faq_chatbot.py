from keras_bert import load_trained_model_from_checkpoint
from keras.layers import Layer
from keras_bert import Tokenizer
import keras as keras
from keras import backend as K
import os
import codecs
import numpy as np

SEQ_LEN = 384

config_path = os.path.join('bert', 'bert_config.json')       
checkpoint_path = os.path.join('bert', 'bert_model.ckpt')    
vocab_path = os.path.join('bert', 'vocab.txt')

model = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=False, trainable=True, seq_len=SEQ_LEN,)

class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def get_output_shape_for(self, input_shape):   
        return input_shape
    
class MyLayer_Start(Layer):

    def __init__(self,seq_len, **kwargs):
        
        self.seq_len = seq_len
        self.supports_masking = True
        super(MyLayer_Start, self).__init__(**kwargs)

    def build(self, input_shape):
        
        self.W = self.add_weight(name='kernel', 
                                 shape=(input_shape[2],2),
                                 initializer='uniform',
                                 trainable=True)
        super(MyLayer_Start, self).build(input_shape)

    def call(self, x):
        
        x = K.reshape(x, shape=(-1,self.seq_len,K.shape(x)[2]))
        x = K.dot(x, self.W)
        
        x = K.permute_dimensions(x, (2,0,1))

        self.start_logits, self.end_logits = x[0], x[1]
        
        self.start_logits = K.softmax(self.start_logits, axis=-1)
        
        return self.start_logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len)


class MyLayer_End(Layer):
    
    def __init__(self,seq_len, **kwargs):
        
        self.seq_len = seq_len
        self.supports_masking = True
        super(MyLayer_End, self).__init__(**kwargs)
  
    def build(self, input_shape):
        
        self.W = self.add_weight(name='kernel', 
                                 shape=(input_shape[2], 2),
                                 initializer='uniform',
                                 trainable=True)
        super(MyLayer_End, self).build(input_shape)

  
    def call(self, x):

        
        x = K.reshape(x, shape=(-1,self.seq_len,K.shape(x)[2]))
        x = K.dot(x, self.W)
        x = K.permute_dimensions(x, (2,0,1))
        
        self.start_logits, self.end_logits = x[0], x[1]
        
        self.end_logits = K.softmax(self.end_logits, axis=-1)
        
        return self.end_logits

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.seq_len)

def get_bert_finetuning_model(model):
    
    inputs = model.inputs[:2]
    dense = model.output
    x = NonMasking()(dense)
    outputs_start = MyLayer_Start(SEQ_LEN)(x)
    outputs_end = MyLayer_End(SEQ_LEN)(x)
    bert_model = keras.models.Model(inputs, [outputs_start, outputs_end])
  
    return bert_model

bert_model = get_bert_finetuning_model(model)
bert_model.load_weights("korquad_3.h5")

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        if "_" in token:
            token = token.replace("_","")
            token = "##" + token
        token_dict[token] = len(token_dict)

reverse_token_dict = {v : k for k, v in token_dict.items()}

class inherit_Tokenizer(Tokenizer):
    def _tokenize(self, text):
        if not self._cased:
            text = text
            
            text = text.lower()
        spaced = ''
        for ch in text:
            if self._is_punctuation(ch) or self._is_cjk_character(ch):
                spaced += ' ' + ch + ' '
            elif self._is_space(ch):
                spaced += ' '
            elif ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        tokens = []
        for word in spaced.strip().split():
            tokens += self._word_piece_tokenize(word)
        return tokens

tokenizer = inherit_Tokenizer(token_dict)

## question과 paragraph를 받아서, token, segment를 만든다.

def convert_pred_data(question, doc):
    global tokenizer
    indices, segments = [], []
    ids, segment = tokenizer.encode(question, doc, max_len=SEQ_LEN)
    indices.append(ids)
    segments.append(segment)
    indices_x = np.array(indices)
    segments = np.array(segments)
    return [indices_x, segments]

def load_pred_data(question, doc):
    data_x = convert_pred_data(question, doc)
    return data_x



# FAQ 답변
def faq_answer(input):
    
    doc, question =input.split(',')
    test_input = load_pred_data(question, doc)             # question과 paragraph를 token,segment로 변수에 저장
    test_start, test_end = bert_model.predict(test_input)  # 학습 모델에 넣고 answer의 start와 end token 예측   
  
    indexes = tokenizer.encode(question, doc, max_len=SEQ_LEN)[0] 
    start = np.argmax(test_start, axis=1).item()      # 예측한 start_token의 위치 
    end = np.argmax(test_end, axis=1).item()          # 예측한 end_token의 위치
    start_tok = indexes[start]                        # 예측한 start_token
    end_tok = indexes[end]                            # 예측한 end_token
    
    sentences = []
    
    for i in range(start, end+1):
        token_based_word = reverse_token_dict[indexes[i]]
        sentences.append(token_based_word)
        #print(token_based_word, end= " ")                   # 예측한 정답, start와 end 토큰 사이의 모든 토큰을 보여줌
    
    answer = []
    for w in sentences:
        if w.startswith("##"):
            w = w.replace("##", "")
        else:
            w = " " + w                                 # 예측한 정답의 ##를 제외하고 보여준다.
        
        answer.append(w)
    
    predict="".join(answer)
    
    return predict