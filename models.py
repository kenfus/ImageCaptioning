import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch

class EncoderCNN(nn.Module):
    def __init__(self, model_param, cnn_model):
        """Load the pretrained cnn_model and replace the top layer."""
        super(EncoderCNN, self).__init__()
        modules = list(cnn_model.children())[:-1]
        self.model = nn.Sequential(*modules)
        self.linear = nn.Linear(list(cnn_model.children())[-1].in_features, model_param['embedding_dim'])
        self.bn = nn.BatchNorm1d(model_param['embedding_dim'])
        
    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.model(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features) # Activation function?
        features = self.bn(features) # Probably dropout not a good idea if only one linear layer; bn also has a regularisation-effect.
        return features


class DecoderRNN(nn.Module):
    def __init__(self, model_param: dict):
        super(DecoderRNN, self).__init__()
        self.sentence_max_length = model_param['sentence_max_length']
        self.token_start = model_param['TOKEN_START']
        self.word_embeddings = nn.Embedding(model_param['vocab_size'], model_param['embedding_dim'])
        self.inv_word_embeddings = nn.Linear(model_param['embedding_dim'], model_param['vocab_size'])

        self.lstm = nn.LSTM(input_size=model_param['embedding_dim'], 
                            hidden_size=model_param['embedding_dim'],
                            batch_first=True) #lstm

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)

    def forward(self, features, captions):
        # Create input:
        inputs = self.word_embeddings(captions[:,0]) # Pass first token, should correspond to model_param[TOKEN_START] in embedding-dimension.

        # Create Output:
        out = []
        out.append(self.inv_word_embeddings(inputs)) # Append TOKEN_START to Output in Vocab-Dimension.

        inputs = inputs.unsqueeze(1)
        features = features.unsqueeze(-3) # Pass features as hidden states?
        hiddens = (torch.zeros_like(features), features)

        # Create Output:
        out = []
        out.append(self.inv_word_embeddings(self.word_embeddings(captions[:,0])))
        
        for i in range(1, self.sentence_max_length):
            outputs, hiddens = self.lstm(inputs, hiddens)
            outputs = self.inv_word_embeddings(outputs.squeeze(1))
            out.append(outputs)
            inputs = self.word_embeddings(captions[:,i])
            inputs = inputs.unsqueeze(1)                         
                                                                  
        return torch.stack(out, 1) 
    
    def test_sample(self, features):
        # Create input:
        batch_size = features.shape[0]
        inputs = self.word_embeddings(self.token_start.repeat(batch_size))

        # Create Output:
        out = []
        out.append(self.inv_word_embeddings(inputs)) # Append TOKEN_START to Output in Vocab-Dimension.

        inputs = inputs.unsqueeze(1) # Add dimension for sentence length. We have one word, thus it should be equal to 1!
        features = features.unsqueeze(-3) # Pass features as hidden states?
        hiddens = (torch.zeros_like(features), features)
        
        for _ in range(1, self.sentence_max_length):
            outputs, hiddens = self.lstm(inputs, hiddens)              
            outputs = self.inv_word_embeddings(outputs.squeeze(1))    
            out.append(outputs)
            inputs = self.word_embeddings(outputs.max(1)[1])           
            inputs = inputs.unsqueeze(1)                               
                                                                       
        return torch.stack(out, 1) 
    
    
    def sample(self, features):
        # Create input:
        batch_size = features.shape[0]
        inputs = self.word_embeddings(self.token_start.repeat(batch_size))

        # Create Output:
        out = []
        out.append(self.inv_word_embeddings(inputs).max(1)[1]) # Append TOKEN_START to Output in Vocab-Dimension.

        inputs = inputs.unsqueeze(1)
        features = features.unsqueeze(-3) # Pass features as hidden states?
        hiddens = (torch.zeros_like(features), features)
        
        for _ in range(1, self.sentence_max_length):
            outputs, hiddens = self.lstm(inputs, hiddens)          
            outputs = self.inv_word_embeddings(outputs.squeeze(1)) 
            out.append(outputs.max(1)[1])
            inputs = self.word_embeddings(out[-1])                 
            inputs = inputs.unsqueeze(1)
                                                                   
        return torch.stack(out, 1) 