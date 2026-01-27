import torch
import torch.nn as nn

class EALSTM(nn.Module):
    def __init__(self, input_dim_dyn, input_dim_stat, hidden_dim, dropout=0.4):
        super(EALSTM, self).__init__()
        self.input_dim_dyn = input_dim_dyn
        self.input_dim_stat = input_dim_stat
        self.hidden_dim = hidden_dim
        
        # EA-LSTM Gate Generators
        # Input Gate (i) depends ONLY on Static Features
        self.input_gate = nn.Linear(input_dim_stat, hidden_dim)
        
        # Forget (f), Cell (g), Output (o) Gates depend on Dynamic Input + Hidden State
        self.dynamic_gates = nn.LSTMCell(input_dim_dyn, hidden_dim)
        
        # Final head to map hidden state to streamflow
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_dyn, x_stat):
        """
        x_dyn: (Batch, Seq_Len, Dyn_Feats)
        x_stat: (Batch, Stat_Feats)
        """
        batch_size, seq_len, _ = x_dyn.size()
        
        # 1. Compute Input Gate (i) - Constant over time for the whole sequence
        # Sigmoid activation to act as a gate (0 to 1)
        i = torch.sigmoid(self.input_gate(x_stat))
        
        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.hidden_dim).to(x_dyn.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x_dyn.device)
        
        # 2. Iterate through time
        for t in range(seq_len):
            x_t = x_dyn[:, t, :]
            
            # Standard LSTM step BUT we overwrite the input gate
            # PyTorch LSTMCell returns (h_next, c_next)
            # Internally it computes i, f, g, o. We need to hack this or manually implement.
            # Manual implementation is cleaner for EA-LSTM:
            
            # (To save space, we use the LSTMCell for the heavy lifting of f, g, o
            # but we discard its calculated 'i' and inject ours)
            
            # Actually, standard LSTMCell doesn't expose gates easily. 
            # Let's do the manual cell implementation for clarity and correctness.
            pass 
            
        # --- RE-IMPLEMENTING MANUAL LOOP FOR CORRECTION ---
        return self.manual_forward(x_dyn, x_stat, i)

    def manual_forward(self, x_dyn, x_stat, i_gate):
        batch_size, seq_len, _ = x_dyn.size()
        
        # Weights for Dynamic parts (Forget, Cell, Output)
        # We define them here if not defined in init, but better to define in init.
        # For simplicity in this snippet, let's assume we defined a standard LSTM
        # and we extract weights.
        
        # Let's use a standard LSTM but modify the input:
        # EA-LSTM Paper strategy: 
        # The input gate is calculated from static, everything else from dynamic.
        
        # Redefine Init for clarity:
        self.lstm = nn.LSTM(self.input_dim_dyn, self.hidden_dim, batch_first=True)
        # But this doesn't allow 'i' injection.
        
        # Proper EA-LSTM Cell Logic:
        # i = sigmoid(W_i * x_stat + b_i)
        # f = sigmoid(W_f * [h_{t-1}, x_t] + b_f)
        # g = tanh(W_g * [h_{t-1}, x_t] + b_g)
        # o = sigmoid(W_o * [h_{t-1}, x_t] + b_o)
        # c_t = f * c_{t-1} + i * g  <-- This is where the magic happens
        # h_t = o * tanh(c_t)
        
        # We need Linear layers for f, g, o
        self.w_f = nn.Linear(self.input_dim_dyn + self.hidden_dim, self.hidden_dim)
        self.w_g = nn.Linear(self.input_dim_dyn + self.hidden_dim, self.hidden_dim)
        self.w_o = nn.Linear(self.input_dim_dyn + self.hidden_dim, self.hidden_dim)
        
        h = torch.zeros(batch_size, self.hidden_dim).to(x_dyn.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x_dyn.device)
        
        for t in range(seq_len):
            x_t = x_dyn[:, t, :]
            combined = torch.cat((x_t, h), dim=1)
            
            f = torch.sigmoid(self.w_f(combined))
            g = torch.tanh(self.w_g(combined))
            o = torch.sigmoid(self.w_o(combined))
            
            # The EA-LSTM modification: i comes from static (passed as arg i_gate)
            c = f * c + i_gate * g
            h = o * torch.tanh(c)
            
        return self.head(self.dropout(h))
    
    # Update __init__ with these layers
    def __init__(self, input_dim_dyn, input_dim_stat, hidden_dim=256, dropout=0.4):
        super().__init__()
        self.input_dim_dyn = input_dim_dyn
        self.hidden_dim = hidden_dim
        
        # Static Input Gate
        self.input_gate_net = nn.Linear(input_dim_stat, hidden_dim)
        
        # Dynamic Gates (Forget, Cell, Output)
        # Input size is Dyn_Feats + Hidden (for concatenation)
        self.w_f = nn.Linear(input_dim_dyn + hidden_dim, hidden_dim)
        self.w_g = nn.Linear(input_dim_dyn + hidden_dim, hidden_dim)
        self.w_o = nn.Linear(input_dim_dyn + hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x_dyn, x_stat):
        batch_size, seq_len, _ = x_dyn.size()
        
        # 1. Compute Input Gate (once per sequence)
        i = torch.sigmoid(self.input_gate_net(x_stat))
        
        # 2. Init State
        h = torch.zeros(batch_size, self.hidden_dim).to(x_dyn.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x_dyn.device)
        
        # 3. Time Loop
        for t in range(seq_len):
            x_t = x_dyn[:, t, :]
            combined = torch.cat((x_t, h), dim=1)
            
            f = torch.sigmoid(self.w_f(combined))
            g = torch.tanh(self.w_g(combined))
            o = torch.sigmoid(self.w_o(combined))
            
            c = f * c + i * g
            h = o * torch.tanh(c)
            
        return self.head(self.dropout(h))
