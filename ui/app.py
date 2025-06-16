import streamlit as st 
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, TransformerConv
import torch_geometric.nn as pyg_nn
import matplotlib.pyplot as plt


class Filter(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 k: int = 3
                 ) -> None:
        super(Filter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = k
        self.layers = nn.ModuleList([TransformerConv(in_channels, out_channels, edge_dim=2) for _ in range(k)])
        self.norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(k)])

    def forward(self, X, edge_index, edge_attr):
        x = X
        s = torch.zeros(X.shape[0], self.out_channels)

        A = to_dense_adj(edge_index)[0]
        for i, layer in enumerate(self.layers):
            s = self.norms[i](s + layer(x, edge_index, edge_attr))
        return torch.relu(s)


class PowerGridGNN(nn.Module):
    def __init__(self, in_features=2, hidden_dim=16, out_features=2):
        super(PowerGridGNN, self).__init__()
        self.conv1 = Filter(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_features)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        vmax, vmin = x[:, 0], x[:, 1]
        x = self.conv1(x, edge_index, edge_attr)  # First GCN layer
        x = self.conv2(x, edge_index)  # Output layer (Voltage predictions)
        x = x.squeeze()  # Return node-wise voltage values
        x[:, 0] = (vmax - vmin) * torch.sigmoid(x[:, 0]) + vmin
        x[:, 1] = 30 * torch.sigmoid(x[:, 1])
        return x
# Initialize and test the model

def extract_data(net : pp.auxiliary.pandapowerNet):
    X = pd.DataFrame()

    Y = pd.DataFrame(columns=['v_kv', 'angle'])
    try:
        coeff = net.bus.max_vm_pu[0], net.bus.min_vm_pu[0]
    except:
        coeff = 1.1, .9

    X['voltage'] = net.bus["vn_kv"]
    X['max_v'] = coeff[0] * X["voltage"]
    X["min_v"] = coeff[1] * X["voltage"]
    X['bus'] = net.bus.name
    X['d_p_mw'] = 0
    X['d_q_mvar'] = 0

    X['g_min_p'] = 0
    X['g_max_p'] = 0

    X['g_min_q'] = 0
    X['g_max_q'] = 0

    for idx, bus in net.load.iterrows():
        bus_name = bus.bus
        if bus_name in net.load['bus'].values:
            X.loc[X['bus'] == bus_name, 'd_p_mw'] = net.load.loc[idx, 'p_mw'] 
            X.loc[X['bus'] == bus_name, 'd_q_mvar'] = net.load.loc[idx, 'q_mvar']

    for idx, bus in net.gen.iterrows():
        bus_name = bus.bus
        if bus_name in net.load['bus'].values:
            X.loc[X['bus'] == bus_name, 'g_max_p'] = net.gen.loc[idx, 'max_p_mw']
            X.loc[X['bus'] == bus_name, 'g_min_p'] = net.gen.loc[idx, 'min_p_mw']
            X.loc[X['bus'] == bus_name, 'g_max_q'] = net.gen.loc[idx, 'max_q_mvar']
            X.loc[X['bus'] == bus_name, 'g_min_q'] = net.gen.loc[idx, 'min_q_mvar']
    

    Y_ij = (1 / (net.line['length_km'] * (net.line["r_ohm_per_km"] + 1j * net.line["x_ohm_per_km"]))).values  # Y = 1/Z
    Y_real = torch.tensor(Y_ij.real, dtype=torch.float32).unsqueeze(1)  # Real part
    Y_imag = torch.tensor(Y_ij.imag, dtype=torch.float32).unsqueeze(1)  # Imaginary part
    edge_attr = torch.cat([Y_real, Y_imag], dim=1).T  # Edge features
    edge_index = torch.tensor(list(zip(net.line["from_bus"], net.line["to_bus"])), dtype=torch.long).T

    pp.runopp(net)
    Y['v_kv'] = net.res_bus["vm_pu"] * net.bus['vn_kv']

    Y['angle'] = net.res_bus["va_degree"]

    X = X.drop(columns=['voltage'])
    X = X.drop(columns=['bus'])
    print(X.columns)
    return torch.from_numpy(X.to_numpy(dtype=np.float32)), edge_attr, edge_index, torch.from_numpy(Y.to_numpy())



st.title("Optimal Power Flow")


cases = []
for i in dir(pn.power_system_test_cases):
    if i.startswith("case"):
        cases.append(i)

with st.container(border=True):
    network = st.selectbox(label='Select test net', options=cases)
    
    btn = st.button('Submit', type='primary')
if btn:
    net = eval(f'pn.{network}()')
    data = extract_data(net)
    X, edge_attr, edge_index, y = data
    y = y.to(torch.float32)
    X = X.to(torch.float32)

    data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr.T, y=y)
    model = PowerGridGNN(in_features=8, hidden_dim=16, out_features=2)
    model.load_state_dict(torch.load('models/model_state.pth', weights_only=True))
    Y = model(data)

    from pandapower.plotting.plotly import create_bus_trace, create_line_trace
    import plotly.graph_objects as go

    bus_traces = create_bus_trace(net, size=10)
    line_traces = create_line_trace(net)

    fig = go.Figure()
    for trace in line_traces:
        fig.add_trace(trace)

    for trace in bus_traces:
        fig.add_trace(trace)

    fig = pp.plotting.simple_plot(net)

    st.pyplot(plt)

    with st.container(border=True):
        st.header("Comparison")
        st.table(pd.DataFrame([
            ['Default', torch.nn.functional.mse_loss(data.y, Y), torch.mean(torch.abs(data.y - Y))]
        ], columns=['Model', 'MSE', 'MAE']))

    display = pd.DataFrame(X.detach().numpy() , columns=['max_v', 'min_v', 'd_p_mw', 'd_q_mvar', 'g_min_p', 'g_max_p', 'g_min_q',
    'g_max_q'])
    output = pd.DataFrame(Y.detach().numpy() , columns=['Voltage (in kv)', 'Angle'])
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("output")
        st.write(output)
    with col2:
        
        st.header("ground Truth")
        y = pd.DataFrame(data.y.numpy(), columns=['Voltage (in kv)', 'Angle'])
        st.write(y)
    

    st.header("Input")
    st.write(display)
    
    
