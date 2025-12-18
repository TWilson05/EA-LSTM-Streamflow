import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false"
tf.config.optimizer.set_jit(False)

# Global
SEQ_LENGTH = 365
MODEL_PATH = 'D:/cxr/UBC/Streamflow_Project/models/ea_lstm_final.keras' 
SCALERS_PATH = 'D:/cxr/UBC/Streamflow_Project/scalers.pkl'
METADATA_PATH = 'D:/cxr/UBC/Streamflow_Project/data/station_data/station_cluster_metadata.csv'

# EA-LSTM
class EntityAwareLSTMLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        dynamic_dim = input_shape[0][-1]
        static_dim = input_shape[1][-1]
        
        self.weight_ih = self.add_weight(shape=(dynamic_dim, 3 * self.units), initializer='orthogonal', name='weight_ih')
        identity = tf.eye(self.units)
        identity_3x = tf.concat([identity, identity, identity], axis=1)
        self.weight_hh = self.add_weight(shape=(self.units, 3 * self.units), initializer=tf.constant_initializer(identity_3x.numpy()), name='weight_hh')
        self.weight_sh = self.add_weight(shape=(static_dim, self.units), initializer='orthogonal', name='weight_sh')
        self.bias = self.add_weight(shape=(3 * self.units,), initializer='zeros', name='bias')
        self.bias_s = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_s')
        self.built = True
    
    def call(self, inputs, training=None):
        x_dynamic, x_static = inputs
        batch_size = tf.shape(x_dynamic)[0]
        seq_len = tf.shape(x_dynamic)[1]
        h = tf.zeros((batch_size, self.units), dtype=x_dynamic.dtype)
        c = tf.zeros((batch_size, self.units), dtype=x_dynamic.dtype)
        i = tf.sigmoid(tf.matmul(x_static, self.weight_sh) + self.bias_s)
        
        def step_fn(t, h, c, h_list):
            x_t = x_dynamic[:, t, :]
            gates = tf.matmul(x_t, self.weight_ih) + tf.matmul(h, self.weight_hh) + self.bias
            f, o, g = tf.split(gates, 3, axis=1)
            c_new = tf.sigmoid(f) * c + i * tf.tanh(g)
            h_new = tf.sigmoid(o) * tf.tanh(c_new)
            h_list = h_list.write(t, h_new)
            return t + 1, h_new, c_new, h_list
        
        h_array = tf.TensorArray(dtype=x_dynamic.dtype, size=seq_len)
        _, h_final, c_final, h_array = tf.while_loop(cond=lambda t, *_: t < seq_len, body=step_fn, loop_vars=[0, h, c, h_array])
        return h_final

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config
    
def calculate_nse(observed, predicted):
        mask = ~(np.isnan(observed) | np.isnan(predicted))
        obs, pred = observed[mask], predicted[mask]
        if len(obs) == 0: return np.nan
        return 1 - (np.sum((obs - pred)**2) / np.sum((obs - np.mean(obs))**2))
    
def load_model_and_scalers(model_path, scalers_path):
    model = keras.models.load_model(model_path, custom_objects={'EntityAwareLSTMLayer': EntityAwareLSTMLayer})
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    return model, scalers

def prepare_sequences(data, station_metadata, scalers, seq_length=365, target_year=None):
    """Prepare sequences for prediction
    
    Parameters:
    -----------
    data : DataFrame
        Full time series data
    station_metadata : Series
        Station metadata
    scalers : dict
        Dictionary of fitted scalers
    seq_length : int
        Length of input sequences (default 365)
    target_year : int or None
        If specified, only return predictions for this year
    
    Returns:
    --------
    X_dynamic, X_static : arrays
        Input sequences
    y_indices : list
        Indices in the original data corresponding to predictions
    dates : DatetimeIndex
        Dates corresponding to predictions
    """
    
    # Calculate area-scaled precipitation
    area = np.exp(station_metadata['log_area'])
    area_scaled_precip = data['precip'].values / area
    
    # Prepare temperature data
    temps = np.column_stack([
        data['max_temp'].values,
        data['min_temp'].values,
        data['mean_temp'].values
    ]).astype(np.float32)
    
    # Normalize temperatures
    temps_scaled = scalers['temp'].transform(temps)
    
    # Normalize area-scaled precipitation
    precip_scaled = scalers['area_scaled_precip'].transform(
        area_scaled_precip.reshape(-1, 1)
    )
    
    # Combine dynamic features
    dynamic_data = np.column_stack([
        temps_scaled[:, 0],  # max_temp
        temps_scaled[:, 1],  # min_temp
        precip_scaled.flatten(),  # area_scaled_precip
        temps_scaled[:, 2]   # mean_temp
    ]).astype(np.float32)
    
    # Prepare static features
    static_features = np.array([
        station_metadata['pct_glaciation'],
        station_metadata['log_area']
    ]).reshape(1, -1).astype(np.float32)
    static_scaled = scalers['static'].transform(static_features)
    
    # Create sequences for all possible dates
    X_dynamic_all = []
    y_indices_all = []
    
    for i in range(len(dynamic_data) - seq_length):
        X_dynamic_all.append(dynamic_data[i:i+seq_length])
        y_indices_all.append(i + seq_length)
    
    # Filter by year if specified
    if target_year is not None:
        # Get indices where prediction date is in target year
        prediction_dates = data.index[y_indices_all]
        year_mask = prediction_dates.year == target_year
        
        # Filter sequences
        X_dynamic = [X_dynamic_all[i] for i in range(len(X_dynamic_all)) if year_mask[i]]
        y_indices = [y_indices_all[i] for i in range(len(y_indices_all)) if year_mask[i]]
        
        if len(X_dynamic) == 0:
            raise ValueError(f"No valid sequences found for year {target_year}")
        
        X_dynamic = np.array(X_dynamic, dtype=np.float16)
    else:
        X_dynamic = np.array(X_dynamic_all, dtype=np.float16)
        y_indices = y_indices_all
    
    # Repeat static features for each sequence
    X_static = np.repeat(static_scaled, len(X_dynamic), axis=0).astype(np.float16)
    
    # Get corresponding dates
    dates = data.index[y_indices]
    
    return X_dynamic, X_static, y_indices, dates

def evaluate_station(model, scalers, station, year=None, seq_length=365, verbose=False):
    DATA_BASE = '/data/wujiaxuan/cxr/data/basin_averaged_climate_data/'
    
    # data
    try:
        max_temp = pd.read_csv(f'{DATA_BASE}basin_max_temperature.csv', index_col=0, parse_dates=True)[str(station)]
        min_temp = pd.read_csv(f'{DATA_BASE}basin_min_temperature.csv', index_col=0, parse_dates=True)[str(station)]
        precip = pd.read_csv(f'{DATA_BASE}basin_total_precipitation.csv', index_col=0, parse_dates=True)[str(station)]
        mean_temp = pd.read_csv(f'{DATA_BASE}basin_mean_temperature.csv', index_col=0, parse_dates=True)[str(station)]
        streamflow = pd.read_csv('/data/wujiaxuan/cxr/data/station_data/combined_streamflow_cleaned.csv', index_col=0, parse_dates=True)[str(station)]
        
        metadata = pd.read_csv(METADATA_PATH).set_index('StationNum')
        station_meta = metadata.loc[int(station) if str(station).isdigit() else station]
    except Exception as e:
        if verbose: print(f"Error loading {station}: {e}")
        return None, None, None, np.nan

    # features
    area = station_meta['Area_km2']
    data = pd.DataFrame({'max_temp': max_temp, 'min_temp': min_temp, 'precip': precip, 'mean_temp': mean_temp, 'streamflow': streamflow})
    X_dynamic, X_static, y_indices, dates = prepare_sequences(data, pd.Series({'pct_glaciation': station_meta['pct_glaciation'], 'log_area': np.log(area)}), scalers, seq_length, target_year=year)
    
    # prediction
    pred_scaled = model.predict([X_dynamic, X_static], batch_size=32, verbose=0)
    pred = scalers['target'][str(station)].inverse_transform(pred_scaled).flatten()
    obs = data['streamflow'].iloc[y_indices].values
    
    return dates, obs, pred, calculate_nse(obs, pred)

import pandas as pd
import numpy as np

# 1. 准备配置
years = range(2007, 2011) 
meta = pd.read_csv('/data/wujiaxuan/cxr/data/station_data/station_cluster_metadata.csv')
stations = meta['StationNum'].astype(str).tolist()

MODEL_PATH = '/data/wujiaxuan/cxr/models/ea_lstm_final.keras'
SCALERS_PATH = '/data/wujiaxuan/cxr/models/scalers.pkl'
SEQ_LENGTH = 365

# 2. 加载模型和缩放器
model, scalers = load_model_and_scalers(MODEL_PATH, SCALERS_PATH)

# 3. 循环计算所有台站的 NSE
results_list = []

print(f"{'StationID':<12} | {'Avg NSE':<10} | {'Glaciation %':<12}")
print("-" * 40)

for station in stations:
    nse_values = []
    # 获取该台站的冰川覆盖率 (pct_glaciation)
    glac_val = meta[meta['StationNum'].astype(str) == station]['pct_glaciation'].values[0]
    
    for year in years:
        try:
            dates, observed, predicted, nse = evaluate_station(
                model, scalers, station, year, SEQ_LENGTH, no_glac=False
            )
            if not np.isnan(nse):
                nse_values.append(nse)
        except Exception as e:
            continue
    
    # 计算该台站在所有年份的平均 NSE
    avg_nse = np.mean(nse_values) if nse_values else np.nan
    
    print(f"{station:<12} | {avg_nse:<10.4f} | {glac_val:<12.2f}%")
    
    results_list.append({
        'StationID': station,
        'Average_NSE': avg_nse,
        'Pct_Glaciation': glac_val
    })

# 4. 转换成 DataFrame 并导出到 CSV 文件
results_df = pd.DataFrame(results_list)
output_file = 'station_nse_summary.csv'
results_df.to_csv(output_file, index=False)

print(f"\n[完成] 所有台站的 NSE 结果已保存至: {output_file}")

import matplotlib.pyplot as plt

df = pd.read_csv('station_nse_summary.csv').dropna()

plt.figure(figsize=(10, 6))
plt.scatter(df['Pct_Glaciation'], df['Average_NSE'], color='royalblue', alpha=0.7, edgecolors='k')

# 添加一条 0.5 的参考线
plt.axhline(y=0.5, color='red', linestyle='--', label='Acceptable (NSE=0.5)')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

plt.xlabel('Percentage of Glaciation (%)', fontsize=12)
plt.ylabel('Average NSE (2007-2010)', fontsize=12)
plt.title('Station Performance (NSE) vs. Basin Glaciation', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.savefig('nse_vs_glaciation.png', dpi=300)
plt.show()