from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import pickle
import os

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.softmax(out)
        return out



def infer_adapter(text, model='${PATH_TO_PRETRAINED_MODEL}/ChartVLM/base/'):
    texts = []
    texts.append(text)
    model_path = os.path.join(model, 'instruction_adapter', 'mlp_classifier.pth')
    tokenizer = os.path.join(model,'instruction_adapter', 'vectorizer.pkl' )

    with open(tokenizer, 'rb') as file:
        vectorizer = pickle.load(file)

    inputs = vectorizer.transform(texts).toarray()

    inputs = torch.tensor(inputs, dtype=torch.float32)

    model = MLPClassifier(input_dim=1719, hidden_dim=512, output_dim=6)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    output = model(inputs)
    _, predicted_label = torch.max(output, dim=1)

    return predicted_label.item()