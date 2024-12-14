import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj_matrix):
        support = torch.matmul(x, self.weight)
        output = torch.matmul(adj_matrix, support) + self.bias
        return output

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_dim, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, output_dim)
    
    def forward(self, x, adj_matrix):
        x = F.relu(self.gc1(x, adj_matrix))
        x = F.dropout(x, training=self.training)
        x = self.gc2(x, adj_matrix)
        return F.log_softmax(x, dim=1)

# Example usage:
# Define your input features, adjacency matrix, and output labels
input_features = torch.randn(5, 3)  # Example: 5 nodes, each with 3 features
adj_matrix = torch.randn(5, 5)  # Example adjacency matrix for 5 nodes
output_labels = torch.LongTensor([0, 1, 2, 3, 4, 5])  # Example output labels

# Create an instance of the GNN model
gnn_model = GNN(input_dim=3, hidden_dim=16, output_dim=6)

# Forward pass
output = gnn_model(input_features, adj_matrix)

print("Output shape:", output.shape)  # Should be (5, 6) for 5 nodes and 6 output classes
