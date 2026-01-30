import torch
import torch.nn as nn

class EALSTM(nn.Module):
    def __init__(self, input_dim_dyn, input_dim_stat, hidden_dim=256, dropout=0.4):
        super(EALSTM, self).__init__()
        self.input_dim_dyn = input_dim_dyn
        self.input_dim_stat = input_dim_stat
        self.hidden_dim = hidden_dim
        
        # 1. Static Input Gate (Calculated once from Basin Attributes)
        self.input_gate_net = nn.Linear(input_dim_stat, hidden_dim)
        
        # 2. Dynamic Gates (Forget, Cell Update, Output)
        # These take [Current Weather + Previous Hidden State]
        combined_dim = input_dim_dyn + hidden_dim
        self.w_f = nn.Linear(combined_dim, hidden_dim)
        self.w_g = nn.Linear(combined_dim, hidden_dim)
        self.w_o = nn.Linear(combined_dim, hidden_dim)
        
        # 3. Output Head
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_dyn, x_stat):
        """
        x_dyn: (Batch, 365, 3) -> [Precip, Tmax, Tmin]
        x_stat: (Batch, 3) -> [Area, Glacier%, Elev]
        """
        batch_size, seq_len, _ = x_dyn.size()
        
        # Step A: Compute the Static Input Gate (i)
        # This acts as a 'filter' for the weather data based on basin properties
        i = torch.sigmoid(self.input_gate_net(x_stat))
        
        # Step B: Initialize hidden (h) and cell (c) states
        h = torch.zeros(batch_size, self.hidden_dim).to(x_dyn.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x_dyn.device)
        
        # Step C: The Temporal Loop (365 days)
        for t in range(seq_len):
            x_t = x_dyn[:, t, :]
            # Concatenate today's weather with yesterday's hidden state
            combined = torch.cat((x_t, h), dim=1)
            
            # Compute dynamic gates
            f = torch.sigmoid(self.w_f(combined))
            g = torch.tanh(self.w_g(combined))
            o = torch.sigmoid(self.w_o(combined))
            
            # Update Cell State (c) and Hidden State (h)
            # This is the core EA-LSTM equation
            c = f * c + i * g
            h = o * torch.tanh(c)
            
        # Step D: Final Prediction for the next day
        return self.head(self.dropout(h))
