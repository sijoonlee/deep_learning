import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) #, dropout = 0.2)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        # captions[:,:-1] # batch_size, caption_length - 1 to exclude <end>
        # -> batch_size, caption_length -1, embed_size
        caption_embed = self.embedding(captions[:,:-1])
       
        # concat img features and caption
        # features.unsqueeze(1) : batch_size, 1, embed_size
        # caption_embed: batch_size, caption_length -1, embed_size
        # -> batch_size, caption_length, embed_size
        combined_embed = torch.cat((features.unsqueeze(1), caption_embed), dim=1) 
        
        output, _ = self.lstm(combined_embed) # "unrolled" connection
        
        output = self.linear(output)
        
        return output
        

    def sample(self, inputs, states=None, max_len=20): # input: 1,1,embed_size
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        list = []
        for i in range(max_len):

            # "rolled" connection to generate one word after one word
            output, states = self.lstm(inputs, states) # output: 1,1,hidden_size
            
            output = self.linear(output) # output: 1,1,vocab_size

            _, top_class = output.topk(1, dim=2)
            
            list.append(top_class.item()) # change type into python int and append it
            
            if(top_class.item() == 1): # if <end> found
                break
            
            inputs = self.embedding(top_class) # 1,1,1,embed_size

            inputs = inputs.squeeze(2) # 1,1,embed_size

        return list