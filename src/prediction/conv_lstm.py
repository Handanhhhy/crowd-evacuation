"""
ConvLSTM网络

用于时空序列预测的卷积LSTM网络。
参考论文: Convolutional LSTM Network (Shi et al., 2015)

输入: [batch, seq_len, 4, 30, 16]
      4通道 = density + vx + vy + exit_distance
输出: [batch, 1, 30, 16] 预测密度
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class ConvLSTMCell(nn.Module):
    """ConvLSTM单元
    
    将LSTM的全连接操作替换为卷积操作，
    能够捕捉空间相关性。
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        """
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏状态通道数
            kernel_size: 卷积核大小
            bias: 是否使用偏置
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 输入门、遗忘门、单元门、输出门 (4个门)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=bias,
        )
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 输入 [batch, input_channels, height, width]
            state: (h, c) 隐藏状态和单元状态
            
        Returns:
            h: 新的隐藏状态
            (h, c): 新的状态元组
        """
        batch, _, height, width = x.shape
        
        if state is None:
            h = torch.zeros(batch, self.hidden_channels, height, width, 
                          device=x.device, dtype=x.dtype)
            c = torch.zeros(batch, self.hidden_channels, height, width,
                          device=x.device, dtype=x.dtype)
        else:
            h, c = state
            
        # 拼接输入和隐藏状态
        combined = torch.cat([x, h], dim=1)
        
        # 卷积计算4个门
        gates = self.conv(combined)
        
        # 分割4个门
        i, f, g, o = torch.chunk(gates, 4, dim=1)
        
        # 门控操作
        i = torch.sigmoid(i)  # 输入门
        f = torch.sigmoid(f)  # 遗忘门
        g = torch.tanh(g)     # 单元门
        o = torch.sigmoid(o)  # 输出门
        
        # 更新单元状态和隐藏状态
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, (h_new, c_new)
    
    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化隐藏状态"""
        h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    """多层ConvLSTM
    
    堆叠多个ConvLSTM层，增强时空建模能力。
    """
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        kernel_size: int = 3,
        num_layers: int = 1,
        batch_first: bool = True,
        return_all_layers: bool = False,
    ):
        """
        Args:
            input_channels: 输入通道数
            hidden_channels: 每层隐藏通道数列表
            kernel_size: 卷积核大小
            num_layers: 层数
            batch_first: 输入是否batch在前
            return_all_layers: 是否返回所有层的输出
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_all_layers = return_all_layers
        
        # 确保hidden_channels是列表
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        
        assert len(hidden_channels) == num_layers
        
        # 创建ConvLSTM层
        cell_list = []
        for i in range(num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_channels=cur_input_channels,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size,
                )
            )
        
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: 输入 [batch, seq_len, channels, height, width]
            hidden_state: 初始隐藏状态列表
            
        Returns:
            output: 输出序列
            last_state: 最后的隐藏状态列表
        """
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)  # (t, b, c, h, w) -> (b, t, c, h, w)
            
        batch, seq_len, _, height, width = x.shape
        
        # 初始化隐藏状态
        if hidden_state is None:
            hidden_state = [
                cell.init_hidden(batch, height, width, x.device)
                for cell in self.cell_list
            ]
            
        # 逐层处理
        layer_output_list = []
        layer_state_list = []
        
        cur_layer_input = x
        
        for layer_idx, cell in enumerate(self.cell_list):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # 逐时间步处理
            for t in range(seq_len):
                h, (h, c) = cell(cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
                
            # 堆叠时间步输出
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            layer_state_list.append((h, c))
            
        if self.return_all_layers:
            return layer_output_list, layer_state_list
        else:
            return layer_output_list[-1], layer_state_list


class DensityPredictorNet(nn.Module):
    """密度场预测网络
    
    基于ConvLSTM的端到端密度预测模型。
    
    输入: [batch, seq_len, 4, 30, 16]
          4通道 = density + vx + vy + exit_distance
    输出: [batch, 1, 30, 16] 预测密度 (归一化到[0,1])
    
    网络结构:
        Encoder: Conv2d(4, 32) -> Conv2d(32, 64)
        ConvLSTM: 64 -> 64
        Decoder: Conv2d(64, 32) -> Conv2d(32, 1) -> Sigmoid
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: int = 64,
        encoder_channels: int = 32,
        kernel_size: int = 3,
        num_lstm_layers: int = 2,
        grid_size: Tuple[int, int] = (30, 16),
    ):
        """
        Args:
            input_channels: 输入通道数 (density + vx + vy + exit_dist)
            hidden_channels: ConvLSTM隐藏通道数
            encoder_channels: 编码器中间通道数
            kernel_size: 卷积核大小
            num_lstm_layers: ConvLSTM层数
            grid_size: 网格尺寸 (width, height)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.grid_size = grid_size
        
        # 编码器：提取空间特征
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels, hidden_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # ConvLSTM：捕捉时空关系
        self.convlstm = ConvLSTM(
            input_channels=hidden_channels,
            hidden_channels=[hidden_channels] * num_lstm_layers,
            kernel_size=kernel_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            return_all_layers=False,
        )
        
        # 解码器：生成预测密度
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, encoder_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(encoder_channels, 1, kernel_size, padding=kernel_size//2),
            nn.Sigmoid(),  # 确保输出在[0, 1]范围
        )
        
    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: 输入序列 [batch, seq_len, 4, height, width]
            hidden_state: 初始隐藏状态
            
        Returns:
            prediction: 预测密度 [batch, 1, height, width]
            last_state: 最后的隐藏状态
        """
        batch, seq_len, channels, height, width = x.shape
        
        # 编码每个时间步
        encoded = []
        for t in range(seq_len):
            enc_t = self.encoder(x[:, t, :, :, :])
            encoded.append(enc_t)
        encoded = torch.stack(encoded, dim=1)  # [batch, seq_len, hidden, h, w]
        
        # ConvLSTM处理时序
        lstm_output, last_state = self.convlstm(encoded, hidden_state)
        
        # 使用最后一个时间步的输出解码
        last_output = lstm_output[:, -1, :, :, :]  # [batch, hidden, h, w]
        
        # 解码生成预测密度
        prediction = self.decoder(last_output)  # [batch, 1, h, w]
        
        return prediction, last_state
    
    def predict_multi_step(
        self,
        x: torch.Tensor,
        steps: int = 1,
        use_teacher_forcing: bool = False,
    ) -> torch.Tensor:
        """多步预测
        
        Args:
            x: 输入序列 [batch, seq_len, 4, height, width]
            steps: 预测步数
            use_teacher_forcing: 是否使用teacher forcing
            
        Returns:
            predictions: 多步预测结果 [batch, steps, 1, height, width]
        """
        predictions = []
        current_input = x
        hidden_state = None
        
        for step in range(steps):
            pred, hidden_state = self.forward(current_input, hidden_state)
            predictions.append(pred)
            
            if step < steps - 1:
                # 构造下一步输入：用预测密度替换当前密度
                # 保持流场和出口距离不变
                last_frame = current_input[:, -1:, :, :, :].clone()
                last_frame[:, :, 0:1, :, :] = pred.unsqueeze(1)  # 替换密度通道
                current_input = torch.cat([current_input[:, 1:, :, :, :], last_frame], dim=1)
        
        return torch.stack(predictions, dim=1)


class DensityPredictorLite(nn.Module):
    """轻量级密度预测网络
    
    适用于实时推理的简化版本，使用更少的参数。
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        grid_size: Tuple[int, int] = (30, 16),
    ):
        super().__init__()
        
        self.grid_size = grid_size
        
        # 简化的编码-解码结构
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
        )
        
        # 单层ConvLSTM
        self.convlstm = ConvLSTMCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size, padding=kernel_size//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 1, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: 输入序列 [batch, seq_len, 4, height, width]
            state: 隐藏状态
            
        Returns:
            prediction: 预测密度 [batch, 1, height, width]
            state: 新的隐藏状态
        """
        batch, seq_len, _, height, width = x.shape
        
        # 逐帧处理
        for t in range(seq_len):
            enc = self.encoder(x[:, t, :, :, :])
            _, state = self.convlstm(enc, state)
        
        h, _ = state
        prediction = self.decoder(h)
        
        return prediction, state
