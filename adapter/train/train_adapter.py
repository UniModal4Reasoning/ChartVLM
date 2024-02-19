# Optimizer for image-to-text task
# Written by Renqiu Xia, Hancheng Ye
# All Rights Reserved 2024-2025.

import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def load_data(file_path):
    texts, labels = [], []
    with open(file_path, 'r') as file:
        for line in file:
            text = line.strip().split('\\t')[0]
            label = line.strip().split('\\t')[-1]
            texts.append(text)
            labels.append(int(label))
    return texts, labels

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to the training data file")
    parser.add_argument("--save_model_path", required=True, help="Path to save the trained model")
    args = parser.parse_args()
    data_path = args.data_path
    save_model_path = args.save_model_path

    texts, labels = load_data(data_path)

    vectorizer = CountVectorizer(max_features=10000) # max words
    X = vectorizer.fit_transform(texts).toarray()


    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)


    train_dataset = TextDataset(X_train, y_train)
    test_dataset = TextDataset(X_test, y_test)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


    input_dim = X_train.shape[1]
    hidden_dim = 512
    output_dim = len(set(labels))
    print(input_dim,output_dim)


    model = MLPClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20

    for epoch in tqdm(range(num_epochs)):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = torch.Tensor(inputs.float())
            labels = torch.LongTensor(labels)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')

    os.makedirs(save_model_path, exist_ok=True)
    torch.save(model.state_dict(), save_model_path + '/mlp_classifier.pth')
    with open(save_model_path + '/vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
    print('Model saved successfully!')


    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = torch.Tensor(inputs.float())
        labels = torch.LongTensor(labels)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test samples: {100 * correct / total}%')