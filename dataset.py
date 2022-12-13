import numpy as np
import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, vocabulary_vectors):
        self.X = X
        self.y = y
        self.vocabulary_vectors = vocabulary_vectors
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __getitem__(self, index):
        sentence = self.X[index]
        sentence = [self.vocabulary_vectors[word_id] if word_id>=0 else np.zeros(100) for word_id in sentence]
        # sentence = np.array(sentence)
        sentence = torch.as_tensor(sentence, dtype=torch.float32, device=self.device)
        label = torch.as_tensor([self.y[index]], dtype=torch.long, device=self.device)
        return sentence, label
    
    def __len__(self):
        return self.X.shape[0]


# print("Dataset爷还没写好呢🤣")