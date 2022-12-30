import math

import torch


class LSTM(torch.nn.Module):
    def __init__(self, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=100, hidden_size=hidden_size, num_layers=1,
                            batch_first=True)
        self.fc = torch.nn.Sequential(torch.nn.Dropout(0.5),
                                      torch.nn.Linear(hidden_size, 32),
                                      torch.nn.Linear(32, 1),
                                      torch.nn.Sigmoid())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, input_seq):
        # print(x.size())
        x, _ = self.lstm(input_seq)
        x = self.fc(x)
        x = x[:, -1, :]
        return x

class MyLSTM(torch.nn.Module):
    def __init__(self, d_model, output_dimension):
        super(MyLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=d_model, hidden_size=256, num_layers=1)
        self.decoder = torch.nn.Linear(256, output_dimension)
        self.sigmoid = torch.nn.Sigmoid()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
        
    def forward(self, x):
        x = x.transpose(0, 1) # [len, batch, d_model]
        h0 = torch.zeros([1, x.shape[1], 256]).to(self.device)
        c0 = torch.zeros([1, x.shape[1], 256]).to(self.device)
        x, (h, c) = self.lstm(x, (h0, c0))
        x = self.decoder(x[-1])
        x = self.sigmoid(x)
        return x



class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        '''对已编码的单词进行位置编码。
        ---
        公式：新的编码 = 原编码+位置编码。
        
        max_len是摆设，不重要，别纠结。d_model是单词embedding维数。
        '''
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MyTransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, output_dimension, num_layers=3, nhead=4):
        '''我自定义的Transformer Encoder。
        ---
        d_model是embedding token的大小，在这里就是我们已有的glove。大小是50或100。
        
        output_dimension是最后的全连接层输出的大小，这里应该是1，因为我们想输出一个二分类结果。
        '''
        super(MyTransformerEncoder, self).__init__()
        self.positional_embedding = PositionalEncoding(d_model)
        layer = torch.nn.TransformerEncoderLayer(d_model, nhead=nhead)
        norm = torch.nn.LayerNorm(d_model)
        self.transformer = torch.nn.TransformerEncoder(layer, num_layers, norm)
        self.linear = torch.nn.Linear(d_model, output_dimension)
        self.sigmoid = torch.nn.Sigmoid()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
    
    def forward(self, x: torch.Tensor):
        '''通过网络。
        ---
        具体而言：
        
        先进行位置编码（因为输入已经是编码过的单词，无需将原单词手动编码）
        
        再经过若干层attention
        
        最后通过一个线性层，输出一个二分类结果。
        '''
        x = x.transpose(0, 1) # [len, batch, d_model]
        x = self.positional_embedding(x)
        x = self.transformer(x)
        x = self.linear(x[-1])
        x = self.sigmoid(x)
        return x

