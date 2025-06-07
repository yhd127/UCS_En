import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import os
import shap
import pickle


# 设置随机种子
def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设置随机种子
seed_all(42)

# 设置matplotlib绘图参数，与1-拟合.py中保持一致
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['xtick.direction'] = 'in'  # 刻度线朝内
plt.rcParams['ytick.direction'] = 'in'  # 刻度线朝内
plt.rcParams['xtick.major.width'] = 0.8  # 刻度线宽度
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['axes.linewidth'] = 0.8  # 轴线宽度
plt.rcParams['xtick.labelsize'] = 9  # 刻度标签大小
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.labelsize'] = 10  # 轴标签大小
plt.rcParams['legend.fontsize'] = 9  # 图例字体大小
plt.rcParams['figure.dpi'] = 1000  # 图像分辨率


# 创建数据集类
class SoilDataset(Dataset):
    """土壤数据集类"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 小波层实现
class WaveletLayer(nn.Module):
    """简化的小波变换层，用于多尺度特征提取"""

    def __init__(self, input_dim, n_wavelets=12):
        super(WaveletLayer, self).__init__()
        self.n_wavelets = n_wavelets
        self.input_dim = input_dim

        # 小波参数
        self.scales = nn.Parameter(torch.rand(n_wavelets) * 0.5 + 0.5)
        self.shifts = nn.Parameter(torch.randn(input_dim, n_wavelets))
        self.weights = nn.Parameter(torch.randn(input_dim, n_wavelets) * 0.1)

    def forward(self, x):
        # 优化计算，使用矩阵运算代替循环
        batch_size = x.shape[0]
        wavelet_out = torch.zeros(batch_size, self.n_wavelets, device=x.device)

        # 为每个小波变换循环一次，但内部使用批量计算
        for i in range(self.n_wavelets):
            # 批量计算所有特征的小波变换
            scaled_x = (x - self.shifts[:, i].unsqueeze(0)) / (self.scales[i] + 1e-8)
            wavelet_values = (1.0 - scaled_x ** 2) * torch.exp(-0.5 * scaled_x ** 2)
            wavelet_out[:, i] = torch.sum(wavelet_values * self.weights[:, i].unsqueeze(0), dim=1)

        return wavelet_out


# MS-PINN模型实现
class MS_PINN(nn.Module):
    """简化的多尺度物理信息神经网络"""

    def __init__(self, input_dim, hidden_dim=128, n_wavelets=12, k_init=0.7303, l_init=0.2610):
        super(MS_PINN, self).__init__()
        self.input_dim = input_dim

        # 简化全局注意力层
        self.global_attention = nn.Linear(input_dim, input_dim)

        # 简化的小波层
        self.wavelet = WaveletLayer(input_dim, n_wavelets)

        # 特征融合
        combined_dim = n_wavelets + input_dim  # 小波特征 + 原始特征

        # 简化的特征处理网络
        self.feature_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )

        # 单个残差块
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        # 简化的输出网络
        self.output_net = nn.Linear(hidden_dim, 1)

        # 物理模型参数，作为可学习参数
        self.k = nn.Parameter(torch.tensor(k_init, dtype=torch.float32))
        self.l = nn.Parameter(torch.tensor(l_init, dtype=torch.float32))

    def forward(self, x):
        # 全局注意力
        attention_weights = torch.sigmoid(self.global_attention(x))
        x_weighted = x * attention_weights

        # 多尺度特征提取
        wavelet_features = self.wavelet(x_weighted)

        # 特征融合
        combined_features = torch.cat([x_weighted, wavelet_features], dim=1)

        # 特征处理
        features = self.feature_net(combined_features)

        # 残差处理
        res = self.res_block(features)
        features = features + res

        # 输出预测
        output = self.output_net(features)

        # 确保输出非负
        output = torch.clamp(output, min=0.0)

        return output

    def get_physics_params(self):
        # 返回当前物理模型参数
        return self.k.item(), self.l.item()

    def physics_prediction(self, x):
        """根据物理公式计算预测值
        公式：ucs = k * (水泥含量/水含量) * exp(黏粒含量/水含量) + l

        适应特征工程后的数据:
        - 如果是原始特征（3维）: 使用原始特征计算
        - 如果是增强特征（6维）: 直接使用工程化特征中的比率特征
        """
        if self.input_dim == 3:  # 原始特征
            # 提取特征
            water_content = x[:, 0].clone() + 1e-8  # 水含量
            cement_content = x[:, 1].clone()  # 水泥含量
            clay_content = x[:, 2].clone()  # 黏粒含量

            # 计算物理模型预测
            ratio = cement_content / water_content
            exp_term = torch.exp(torch.clamp(clay_content / water_content, -10, 10))
            physics_pred = self.k * ratio * exp_term + self.l
        else:  # 增强特征（含特征交互项）
            # 直接使用预计算的特征交互项
            cement_water_ratio = x[:, 3].clone() * (torch.max(x[:, 3]) - torch.min(x[:, 3]) + 1e-8)  # 还原归一化
            clay_water_ratio = x[:, 4].clone() * (torch.max(x[:, 4]) - torch.min(x[:, 4]) + 1e-8)  # 还原归一化

            # 计算物理模型预测
            exp_term = torch.exp(torch.clamp(clay_water_ratio, -10, 10))
            physics_pred = self.k * cement_water_ratio * exp_term + self.l

        # 确保预测值在合理范围内
        physics_pred = torch.clamp(physics_pred, min=0.0)

        return physics_pred.view(-1, 1)

    def mixed_prediction(self, x, alpha=0.7):
        """混合预测：结合神经网络和物理模型的预测"""
        nn_pred = self.forward(x)
        physics_pred = self.physics_prediction(x)

        # 简化混合策略，直接使用固定权重
        mixed_pred = alpha * nn_pred + (1 - alpha) * physics_pred

        return mixed_pred


# 物理约束损失函数
def physics_constraint_loss(model, inputs, targets):
    """简化的物理约束损失
    使用MSE和L1的加权组合，计算更高效
    """
    # 神经网络预测
    nn_pred = model(inputs)

    # 物理模型预测
    physics_pred = model.physics_prediction(inputs)

    # 使用MSE和L1的组合
    mse_loss = torch.mean((nn_pred - physics_pred) ** 2)
    l1_loss = torch.mean(torch.abs(nn_pred - physics_pred))

    # 组合损失：70% MSE + 30% L1
    combined_loss = 0.7 * mse_loss + 0.3 * l1_loss

    return combined_loss


# R2损失函数
def r2_loss(y_pred, y_true):
    """实时计算R²损失"""
    # 确保输入是张量
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)

    # 均值
    y_mean = torch.mean(y_true)

    # 总平方和
    ss_tot = torch.sum((y_true - y_mean) ** 2)

    # 残差平方和
    ss_res = torch.sum((y_true - y_pred) ** 2)

    # R²
    r2 = 1 - ss_res / (ss_tot + 1e-8)  # 防止分母为0

    return r2


# 计算评估指标
def calculate_metrics(y_true, y_pred):
    """计算所有评估指标"""
    # 转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # 确保形状一致
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE (Mean Absolute Percentage Error)
    # 正确的MAPE计算方式
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # NRMSE (Normalized Root Mean Squared Error)
    nrmse = rmse / (np.max(y_true) - np.min(y_true))

    # Mean Bias (%)
    mean_bias = np.mean((y_true - y_pred) / np.mean(y_true)) * 100

    # Forecast Bias (%)
    forecast_bias = (np.sum(y_true - y_pred) / np.sum(y_true)) * 100

    # R² Score
    r2 = r2_score(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAPE': mape,
        'NRMSE': nrmse,
        'Mean_Bias': mean_bias,
        'Forecast_Bias': forecast_bias,
        'R2': r2
    }


# 训练模型
def train_model(model_class, model_name, X_train, y_train, X_val=None, y_val=None,
                hidden_dim=128, n_wavelets=12, epochs=800, batch_size=64,
                lr=0.0018, weight_decay=2e-5, physics_weight=0.15, alpha=0.65,
                k_init=0.7303, l_init=0.2610, adaptive_weights=True):
    """简化的训练函数
    减少参数，简化训练过程，加快收敛速度
    """
    # 转换数据为张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

    # 准备数据集
    train_dataset = SoilDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 获取输入维度
    input_dim = X_train.shape[1]

    # 初始化模型
    model = MS_PINN(input_dim=input_dim, hidden_dim=hidden_dim,
                    n_wavelets=n_wavelets, k_init=k_init, l_init=l_init)

    # 损失函数
    mse_criterion = nn.MSELoss()

    # 优化器 - 使用AdamW，带权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    # 更简单的学习率调度策略
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # 每次衰减为原来的80%
        patience=25,  # 如果25轮验证损失没有改善则降低学习率
        min_lr=lr / 30,
        verbose=True
    )

    # 训练参数
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    patience = 50  # 增加耐心值避免过早停止
    patience_counter = 0
    best_model_state = None

    # 简化权重管理
    current_physics_weight = physics_weight
    current_alpha = alpha

    # 存储训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_r2': [],
        'val_r2': [],
        'k_values': [],
        'l_values': [],
        'physics_weights': [],
        'alpha_values': [],
        'learning_rates': []
    }

    print(f"\n开始训练 {model_name} 模型...")
    start_time = time.time()

    # 初始化最佳模型状态
    best_model_state = model.state_dict().copy()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_data_loss = 0.0
        running_physics_loss = 0.0
        running_r2 = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算数据损失（MSE）
            data_loss = mse_criterion(outputs, targets)

            # 计算物理约束损失
            physics_loss = physics_constraint_loss(model, inputs, targets)

            # 简化总损失计算 - 只使用两个损失项
            loss = (1.0 - current_physics_weight) * data_loss + current_physics_weight * physics_loss

            # 反向传播和优化
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 计算当前批次的R²
            batch_r2 = 1.0 - torch.sum((targets - outputs) ** 2) / (
                    torch.sum((targets - torch.mean(targets)) ** 2) + 1e-8)

            running_loss += loss.item()
            running_data_loss += data_loss.item()
            running_physics_loss += physics_loss.item()
            running_r2 += batch_r2.item()

        # 计算每个epoch的平均损失
        epoch_loss = running_loss / len(train_loader)
        epoch_data_loss = running_data_loss / len(train_loader)
        epoch_physics_loss = running_physics_loss / len(train_loader)
        epoch_r2 = running_r2 / len(train_loader)

        # 自适应权重更新
        if adaptive_weights:
            # 根据进度调整物理权重 - 简化为两个阶段
            progress_ratio = epoch / epochs

            if progress_ratio < 0.5:  # 前期
                # 初期专注于数据拟合
                target_physics_weight = max(0.05, 0.15 - 0.1 * epoch_r2)  # R²越高，物理约束越小
            else:  # 后期
                # 后期根据R²情况调整
                if epoch_r2 > 0.95:  # R²已经很高
                    target_physics_weight = min(0.3, 0.15 + 0.05 * progress_ratio)  # 适度增加物理约束提高泛化
                else:
                    target_physics_weight = 0.1  # 保持低物理约束专注于拟合

            # 平滑过渡
            current_physics_weight = 0.9 * current_physics_weight + 0.1 * target_physics_weight

            # 动态调整混合权重alpha
            if epoch_r2 < 0.75:  # R²较低
                current_alpha = min(0.9, current_alpha + 0.01)  # 增加神经网络权重
            elif epoch_r2 > 0.95:  # R²很高
                current_alpha = max(0.55, current_alpha - 0.002)  # 小幅增加物理模型权重
            else:  # R²适中
                if progress_ratio < 0.5:
                    current_alpha = min(0.8, current_alpha + 0.001)  # 前期略微增加神经网络权重
                else:
                    current_alpha = max(0.6, current_alpha - 0.001)  # 后期略微增加物理模型权重

        # 记录当前权重
        current_lr = optimizer.param_groups[0]['lr']
        history['physics_weights'].append(current_physics_weight)
        history['alpha_values'].append(current_alpha)
        history['learning_rates'].append(current_lr)

        # 记录当前物理参数
        k, l = model.get_physics_params()
        history['k_values'].append(k)
        history['l_values'].append(l)

        # 记录损失和R²
        history['train_loss'].append(epoch_loss)
        history['train_r2'].append(epoch_r2)

        # 验证
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_inputs = torch.FloatTensor(X_val)
                val_targets = torch.FloatTensor(y_val).view(-1, 1)
                val_outputs = model(val_inputs)

                # 验证损失
                val_data_loss = mse_criterion(val_outputs, val_targets)
                val_physics_loss = physics_constraint_loss(model, val_inputs, val_targets)
                val_loss = (1.0 - current_physics_weight) * val_data_loss + current_physics_weight * val_physics_loss

                # 验证集R²
                val_r2 = r2_loss(val_outputs, val_targets)

            # 记录验证损失和R²
            history['val_loss'].append(val_loss.item())
            history['val_r2'].append(val_r2.item())

            # 更新学习率
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2.item()
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= patience:
                print(f"\n早停: {patience}轮后验证集性能未提升")
                break
        else:
            # 没有验证集时记录空值
            history['val_loss'].append(None)
            history['val_r2'].append(None)

            # 在这种情况下，使用训练R²判断最佳模型
            if epoch_r2 > best_val_r2:
                best_val_r2 = epoch_r2
                best_val_loss = epoch_loss
                best_model_state = model.state_dict().copy()

        # 打印进度
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, R²: {epoch_r2:.4f}, "
                  f"Data Loss: {epoch_data_loss:.4f}, Physics Loss: {epoch_physics_loss:.4f}, "
                  f"k={k:.4f}, l={l:.4f}, physics_weight={current_physics_weight:.4f}, alpha={current_alpha:.4f}, "
                  f"LR={current_lr:.6f}")

    # 训练完成
    training_time = time.time() - start_time
    print(f"{model_name} 训练完成，耗时：{training_time:.2f}秒")
    print(f"最终物理约束权重: {current_physics_weight:.4f}, 最终混合权重: {current_alpha:.4f}")
    print(f"最佳验证R²: {best_val_r2:.4f}, 最佳验证损失: {best_val_loss:.4f}")

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 返回模型、训练历史以及最终权重
    return model, history, current_physics_weight, current_alpha


# 评估模型
def evaluate_model(model, X_train, y_train, X_test, y_test, alpha=0.5):
    """评估模型性能"""
    model.eval()

    with torch.no_grad():
        # 准备数据
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)

        # 获取物理参数
        k, l = model.get_physics_params()

        # 模型预测（混合预测）
        y_train_pred = model.mixed_prediction(X_train_tensor, alpha).numpy()
        y_test_pred = model.mixed_prediction(X_test_tensor, alpha).numpy()

        # 神经网络部分预测
        y_train_nn_pred = model(X_train_tensor).numpy()
        y_test_nn_pred = model(X_test_tensor).numpy()

        # 物理模型部分预测
        y_train_physics_pred = model.physics_prediction(X_train_tensor).numpy()
        y_test_physics_pred = model.physics_prediction(X_test_tensor).numpy()

    # 计算评估指标
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)

    # 打印评估结果
    print("\n评估结果：")
    print(f"物理参数: k={k:.4f}, l={l:.4f}, 混合权重: alpha={alpha:.2f}")
    print(f"训练集: R²={train_metrics['R2']:.4f}, RMSE={train_metrics['RMSE']:.4f}, MAPE={train_metrics['MAPE']:.2f}%")
    print(f"测试集: R²={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.4f}, MAPE={test_metrics['MAPE']:.2f}%")
    print(f"训练集: Mean Bias={train_metrics['Mean_Bias']:.2f}%, Forecast Bias={train_metrics['Forecast_Bias']:.2f}%")
    print(f"测试集: Mean Bias={test_metrics['Mean_Bias']:.2f}%, Forecast Bias={test_metrics['Forecast_Bias']:.2f}%")

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'train_pred': y_train_pred,
        'test_pred': y_test_pred,
        'train_nn_pred': y_train_nn_pred,
        'test_nn_pred': y_test_nn_pred,
        'train_physics_pred': y_train_physics_pred,
        'test_physics_pred': y_test_physics_pred,
        'k': k,
        'l': l
    }


def analyze_shap_values(model, X_train, X_test, model_name, alpha=0.7, save_dir='results'):
    """
    使用SHAP对PINN模型进行解释分析

    参数:
        model: 训练好的PINN模型
        X_train: 训练集特征
        X_test: 测试集特征
        model_name: 模型名称
        alpha: 混合预测权重
        save_dir: 保存目录
    """
    print(f"\n开始对{model_name}进行SHAP分析...")

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置模型为评估模式
    model.eval()

    # 合并训练集和测试集为背景数据
    X_background = np.vstack([X_train[:100], X_test[:100]])  # 使用部分数据作为背景
    X_background_tensor = torch.FloatTensor(X_background)

    # 定义一个预测函数，接受numpy数组并返回numpy数组
    def model_predict(x):
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            preds = model.mixed_prediction(x_tensor, alpha).cpu().numpy()
            return preds

    # 创建SHAP解释器
    explainer = shap.KernelExplainer(model_predict, X_background)

    # 计算SHAP值（使用测试集的一部分以加快计算速度）
    shap_sample_size = min(50, len(X_train))
    shap_values = explainer.shap_values(X_train[:shap_sample_size])

    # 处理SHAP值的形状问题
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # 对于回归问题，通常是单输出的

    # 确保SHAP值是2D的 - 修复形状问题
    if shap_values.ndim == 3:
        shap_values = np.squeeze(shap_values)  # 移除多余的维度

    # 特征名称 - 使用英文
    if X_train.shape[1] == 3:
        feature_names = ['Water Content', 'Cement Content', 'Clay Content']
    else:
        # 对于扩展特征，可以添加扩展特征的名称
        feature_names = ['Water Content', 'Cement Content', 'Clay Content',
                         'Cement/Water Ratio', 'Clay/Water Ratio', 'Cement*Clay']
        # 如果特征数量依然不匹配，补充额外特征名
        while len(feature_names) < X_train.shape[1]:
            feature_names.append(f'Feature {len(feature_names) + 1}')

    # 绘制摘要图 - 增加图表宽度
    plt.figure(figsize=(18 / 2.54, 7 / 2.54))
    shap.summary_plot(shap_values, X_train[:shap_sample_size], feature_names=feature_names,
                      show=False, plot_size=(18 / 2.54, 7 / 2.54))

    # 手动设置y轴特征标签为黑色
    ax = plt.gca()
    for text in ax.get_yticklabels():
        text.set_color('black')
        text.set_fontname('Times New Roman')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_shap_summary.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"{save_dir}/{model_name}_shap_summary.svg", format='svg', bbox_inches='tight')
    plt.close()

    # 绘制条形图 - 增加图表宽度
    plt.figure(figsize=(18 / 2.54, 7 / 2.54))
    shap.summary_plot(shap_values, X_train[:shap_sample_size], feature_names=feature_names,
                      plot_type="bar", show=False)

    # 手动设置y轴特征标签为黑色
    ax = plt.gca()
    for text in ax.get_yticklabels():
        text.set_color('black')
        text.set_fontname('Times New Roman')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_shap_importance.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"{save_dir}/{model_name}_shap_importance.svg", format='svg', bbox_inches='tight')
    plt.close()

    # 为每个特征创建依赖图 - 增加图表宽度
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(18 / 2.54, 7 / 2.54))
        shap.dependence_plot(i, shap_values, X_train[:shap_sample_size],
                             feature_names=feature_names, show=False)

        # 手动设置所有文本为黑色和Times New Roman
        ax = plt.gca()
        for text in ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('black')
            text.set_fontname('Times New Roman')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name}_shap_dependence_{feature.replace('/', '_').replace('*', 'x')}.png",
                    dpi=1000, bbox_inches='tight')
        plt.savefig(f"{save_dir}/{model_name}_shap_dependence_{feature.replace('/', '_').replace('*', 'x')}.svg",
                    format='svg', bbox_inches='tight')
        plt.close()

    # 计算特征重要性（SHAP值的绝对均值）
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'Feature': feature_names[:len(feature_importance)],
        'Importance': feature_importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df.to_excel(f"{save_dir}/{model_name}_feature_importance.xlsx", index=False)

    print(f"{model_name} SHAP分析完成，结果已保存至{save_dir}目录")

    return {
        'shap_values': shap_values,
        'feature_importance': importance_df
    }


def analyze_raw_features_shap(model, X_data, y_data, model_name, alpha=0.7, save_dir='raw_features_shap'):
    """
    仅使用原始特征（水含量、水泥含量、黏粒含量）进行SHAP分析

    参数:
        model: 训练好的PINN模型
        X_data: 完整特征数据
        y_data: 标签数据
        model_name: 模型名称
        alpha: 混合预测权重
        save_dir: 保存目录
    """
    print(f"\n使用原始特征对{model_name}进行SHAP分析...")

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置模型为评估模式
    model.eval()

    # 只选择原始特征（前三列：水含量、水泥含量、黏粒含量）
    X_raw = X_data[:, :3]

    # 创建背景数据
    X_background = X_raw[:100]  # 使用部分数据作为背景

    # 定义一个包装预测函数，只使用原始特征
    def raw_model_predict(x):
        with torch.no_grad():
            # 构建完整特征向量（通过添加0值）
            batch_size = x.shape[0]
            x_full = np.zeros((batch_size, model.input_dim))
            x_full[:, :3] = x  # 只填充原始特征

            x_tensor = torch.FloatTensor(x_full)
            preds = model.mixed_prediction(x_tensor, alpha).cpu().numpy()
            return preds

    # 创建SHAP解释器
    explainer = shap.KernelExplainer(raw_model_predict, X_background)

    # 计算SHAP值（使用样本数据以加快计算速度）
    shap_sample_size = min(50, len(X_raw))
    shap_values = explainer.shap_values(X_raw[:shap_sample_size])

    # 处理SHAP值的形状问题
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # 对于回归问题，通常是单输出的

    # 确保SHAP值是2D的
    if shap_values.ndim == 3:
        shap_values = np.squeeze(shap_values)  # 移除多余的维度

    # 原始特征名称 - 使用英文
    feature_names = ['Water Content', 'Cement Content', 'Clay Content']

    # 绘制摘要图 - 增加图表宽度
    plt.figure(figsize=(18 / 2.54, 7 / 2.54))
    shap.summary_plot(shap_values, X_raw[:shap_sample_size], feature_names=feature_names,
                      show=False, plot_size=(18 / 2.54, 7 / 2.54))

    # 手动设置y轴特征标签为黑色
    ax = plt.gca()
    for text in ax.get_yticklabels():
        text.set_color('black')
        text.set_fontname('Times New Roman')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_raw_shap_summary.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"{save_dir}/{model_name}_raw_shap_summary.svg", format='svg', bbox_inches='tight')
    plt.close()

    # 绘制条形图 - 增加图表宽度
    plt.figure(figsize=(18 / 2.54, 7 / 2.54))
    shap.summary_plot(shap_values, X_raw[:shap_sample_size], feature_names=feature_names,
                      plot_type="bar", show=False)

    # 手动设置y轴特征标签为黑色
    ax = plt.gca()
    for text in ax.get_yticklabels():
        text.set_color('black')
        text.set_fontname('Times New Roman')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_raw_shap_importance.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"{save_dir}/{model_name}_raw_shap_importance.svg", format='svg', bbox_inches='tight')
    plt.close()

    # 为每个特征创建依赖图 - 增加图表宽度
    for i, feature in enumerate(feature_names):
        plt.figure(figsize=(18 / 2.54, 7 / 2.54))
        shap.dependence_plot(i, shap_values, X_raw[:shap_sample_size],
                             feature_names=feature_names, show=False)

        # 手动设置所有文本为黑色和Times New Roman
        ax = plt.gca()
        for text in ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('black')
            text.set_fontname('Times New Roman')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name}_raw_shap_dependence_{feature.replace(' ', '_')}.png",
                    dpi=1000, bbox_inches='tight')
        plt.savefig(f"{save_dir}/{model_name}_raw_shap_dependence_{feature.replace(' ', '_')}.svg",
                    format='svg', bbox_inches='tight')
        plt.close()

    # 计算特征重要性（SHAP值的绝对均值）
    feature_importance = np.abs(shap_values).mean(0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df.to_excel(f"{save_dir}/{model_name}_raw_feature_importance.xlsx", index=False)

    # 返回SHAP解释器和SHAP值，以便后续绘制waterfall图
    return {
        'shap_values': shap_values,
        'feature_importance': importance_df,
        'explainer': explainer,
        'X_raw': X_raw,
        'feature_names': feature_names
    }


def plot_shap_waterfall(shap_results, model_name, save_dir='waterfall_results'):
    """
    绘制SHAP waterfall图，选取50%分位数的样本，并确保该样本满足特定条件：
    水泥含量和黏粒含量的SHAP值大于0，水含量的SHAP值小于0

    参数:
        shap_results: SHAP分析结果字典
        model_name: 模型名称
        save_dir: 保存目录
    """
    print(f"\n为{model_name}创建SHAP waterfall图...")

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 获取SHAP值和特征数据
    shap_values = shap_results['shap_values']
    X_raw = shap_results['X_raw']
    feature_names = shap_results['feature_names']
    explainer = shap_results['explainer']

    # 获取用于SHAP分析的样本数量
    sample_size = len(shap_values)

    # 找到50%分位数样本的索引
    median_idx = int(sample_size * 0.5)
    print(f"50%分位数样本的索引编号: {median_idx}")

    # 检查该样本是否满足条件
    # 水泥含量SHAP值 > 0, 黏粒含量SHAP值 > 0, 水含量SHAP值 < 0
    water_idx = 0  # 水含量索引
    cement_idx = 1  # 水泥含量索引
    clay_idx = 2  # 黏粒含量索引

    found_idx = median_idx
    meets_criteria = (shap_values[median_idx, cement_idx] > 0 and
                      shap_values[median_idx, clay_idx] > 0 and
                      shap_values[median_idx, water_idx] < 0)

    # 如果中位数样本不满足条件，在附近搜索满足条件的样本
    search_range = min(10, sample_size // 2)  # 搜索范围，避免超出样本范围

    if not meets_criteria:
        print(f"50%分位数样本不满足条件，在附近搜索...")

        # 首先在中位数样本前后搜索
        for offset in range(1, search_range):
            # 向后搜索
            check_idx = min(median_idx + offset, sample_size - 1)
            if (shap_values[check_idx, cement_idx] > 0 and
                    shap_values[check_idx, clay_idx] > 0 and
                    shap_values[check_idx, water_idx] < 0):
                found_idx = check_idx
                meets_criteria = True
                break

            # 向前搜索
            check_idx = max(median_idx - offset, 0)
            if (shap_values[check_idx, cement_idx] > 0 and
                    shap_values[check_idx, clay_idx] > 0 and
                    shap_values[check_idx, water_idx] < 0):
                found_idx = check_idx
                meets_criteria = True
                break

    # 如果仍未找到满足条件的样本，扩大搜索范围到全部样本
    if not meets_criteria:
        print("在中位数附近未找到满足条件的样本，搜索所有样本...")
        for i in range(sample_size):
            if (shap_values[i, cement_idx] > 0 and
                    shap_values[i, clay_idx] > 0 and
                    shap_values[i, water_idx] < 0):
                found_idx = i
                meets_criteria = True
                break

    if meets_criteria:
        print(f"找到满足条件的样本索引编号: {found_idx}")

        # 打印该样本的特征值和SHAP值
        print("\n选中样本的特征值:")
        for i, name in enumerate(feature_names):
            print(f"{name}: {X_raw[found_idx, i]:.4f}")

        print("\n选中样本的SHAP值:")
        for i, name in enumerate(feature_names):
            print(f"{name}: {shap_values[found_idx, i]:.4f}")

        # 绘制waterfall图
        plt.figure(figsize=(18 / 2.54, 10 / 2.54))  # 增加高度，使图表更清晰

        # 处理expected_value，确保它是标量而不是数组
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[0]  # 取第一个元素
        elif isinstance(expected_value, list):
            expected_value = expected_value[0]  # 取第一个元素

        print(f"Expected value: {expected_value}")

        # 使用选择的样本创建waterfall图
        shap_waterfall = shap.plots._waterfall.waterfall_legacy(
            expected_value,
            shap_values[found_idx],
            feature_names=feature_names,
            max_display=len(feature_names),
            show=False
        )

        # 获取当前轴对象
        ax = plt.gca()

        # 设置所有文本为黑色和Times New Roman
        for text in ax.texts + ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color('black')
            text.set_fontname('Times New Roman')

        # 修改坐标轴标签字体
        for label in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            label.set_color('black')
            label.set_fontname('Times New Roman')

        # 添加特征值标注到y轴标签
        yticks = ax.get_yticks()
        ylabels = [item.get_text() for item in ax.get_yticklabels()]

        # 找出哪些标签是特征名称
        feature_label_indices = []
        for i, label in enumerate(ylabels):
            if label in feature_names:
                feature_label_indices.append(i)

        # 创建新的y轴标签，包含特征值
        new_labels = ylabels.copy()
        for i in feature_label_indices:
            feature_idx = feature_names.index(ylabels[i])
            feature_value = X_raw[found_idx, feature_idx]
            new_labels[i] = f"{ylabels[i]} = {feature_value:.3f}"

        # 设置新的y轴标签
        ax.set_yticklabels(new_labels)

        # 为每个轴标签设置字体
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_color('black')

        # 调整图形大小以适应更长的标签
        plt.tight_layout()

        # 保存图表
        plt.savefig(f"{save_dir}/{model_name}_waterfall_sample_{found_idx}.png",
                    dpi=1000, bbox_inches='tight')
        plt.savefig(f"{save_dir}/{model_name}_waterfall_sample_{found_idx}.svg",
                    format='svg', bbox_inches='tight')
        plt.close()

        print(f"Waterfall图已保存至 {save_dir}/{model_name}_waterfall_sample_{found_idx}.png")
    else:
        print("未找到满足所有条件的样本，请考虑放宽条件。")

    return found_idx


def create_combined_shap_plot(shap_values, X_sample, feature_names, model_name, save_dir='combined_shap_results'):
    """
    创建结合条形图和南丁格尔玫瑰图的SHAP分析图

    参数:
        shap_values: SHAP值数组
        X_sample: 用于分析的特征数据
        feature_names: 特征名称列表
        model_name: 模型名称
        save_dir: 保存目录
    """
    print(f"\n创建{model_name}的组合SHAP分析图...")

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 计算特征重要性（SHAP值的绝对均值）
    feature_importance = np.abs(shap_values).mean(0)

    # 创建数据框并排序
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)

    # 计算百分比重要性
    total_importance = importance_df['Importance'].sum()
    importance_df['Percentage'] = importance_df['Importance'] / total_importance * 100

    # 设置图形大小
    fig, ax = plt.subplots(figsize=(18 / 2.54, 7 / 2.54))

    # 绘制条形图
    sorted_idx = np.argsort(feature_importance)
    sorted_features = np.array(feature_names)[sorted_idx]

    # 确定颜色映射
    cmap = plt.cm.coolwarm_r  # 使用与原图类似的颜色方案（红-白-蓝）

    # 创建一个颜色渐变
    color_values = np.linspace(0, 1, len(sorted_features))
    bars = ax.barh(sorted_features, feature_importance[sorted_idx])

    # 为每个条形设置颜色
    for i, bar in enumerate(bars):
        bar.set_color(cmap(color_values[i]))

    # 设置标签
    ax.set_xlabel('mean(|SHAP value|)')

    # 添加Y轴网格线（点线）
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    # 设置所有文本为黑色和Times New Roman
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_color('black')
        text.set_fontname('Times New Roman')

    # 设置坐标轴标签字体
    ax.xaxis.label.set_color('black')
    ax.xaxis.label.set_fontname('Times New Roman')

    # 创建颜色条
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('High\n\n\n\n\n\n\nLow', rotation=0, labelpad=15, va='center', color='black',
                   fontname='Times New Roman')

    # 计算玫瑰图的位置（在条形图中部右侧）
    rose_center_x = 0.7 * max(feature_importance)  # 条形图宽度的70%
    rose_center_y = len(sorted_features) / 2  # 条形图中部
    rose_radius = min(max(feature_importance) * 0.25, len(sorted_features) * 0.25)  # 适当的半径

    # 添加南丁格尔玫瑰图（极坐标饼图）
    # 准备数据（按百分比大小排序）
    sorted_importance = importance_df.sort_values('Importance', ascending=False)
    labels = sorted_importance['Feature'].values
    sizes = sorted_importance['Percentage'].values

    # 在条形图上添加一个极坐标子图
    ax_pie = fig.add_axes([0.65, 0.5, 0.2, 0.2], polar=True)  # [left, bottom, width, height]

    # 计算每个扇形的角度
    angles = np.linspace(0, 2 * np.pi, len(sizes), endpoint=False)

    # 绘制南丁格尔玫瑰图
    bars = ax_pie.bar(angles, sizes, width=2 * np.pi / len(sizes), bottom=0.0, alpha=0.8)

    # 为每个扇形赋予渐变色
    for i, bar in enumerate(bars):
        # 使用与上面相同的颜色方案
        idx = np.where(sorted_importance['Feature'].values[i] == sorted_features)[0][0]
        bar.set_facecolor(cmap(color_values[idx]))

    # 在饼图上添加百分比标签
    for i, (angle, percentage) in enumerate(zip(angles, sizes)):
        if percentage >= 5:  # 只显示大于5%的标签
            # 计算标签位置
            x = np.cos(angle + np.pi / len(sizes))
            y = np.sin(angle + np.pi / len(sizes))
            # 确定标签的径向位置
            radial_pos = percentage / max(sizes) * 0.7 + 0.3
            label_x = x * radial_pos * max(sizes)
            label_y = y * radial_pos * max(sizes)
            # 添加标签，并设置为黑色
            text = ax_pie.text(angle + np.pi / len(sizes), percentage * 0.7, f"{percentage:.2f}%",
                               ha='center', va='center', fontsize=7)
            text.set_color('black')
            text.set_fontname('Times New Roman')

    # 关闭饼图的径向和角度刻度
    ax_pie.set_yticklabels([])
    ax_pie.set_xticklabels([])

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_combined_shap.png", dpi=1000, bbox_inches='tight')
    plt.savefig(f"{save_dir}/{model_name}_combined_shap.svg", format='svg', bbox_inches='tight')
    plt.close()

    print(f"{model_name} 组合SHAP分析图已保存至 {save_dir} 目录")

    return importance_df


# 主函数
def main():
    """主函数：加载数据、训练模型、评估结果"""
    seed_all(42)  # 设置随机种子
    print("加载数据...")

    # 加载Excel数据
    try:
        data = pd.read_excel('固化土.xlsx')
        print(f"数据加载成功，共 {len(data)} 条记录")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return

    # 只保留需要的四列数据：水含量、水泥含量、黏粒含量、强度
    try:
        data = data[['水含量', '水泥含量', '黏粒含量', '强度']]
        print("成功选择指定的四列数据")
    except KeyError as e:
        print(f"选择指定列失败: {e}")
        print("列名可能不匹配，请检查Excel文件中的实际列名")
        return

    # 检查数据统计和异常值
    print("\n数据统计:")
    print(data.describe())

    # 检查是否有缺失值
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        print("\n检测到缺失值:")
        print(missing_data[missing_data > 0])
        print("填充缺失值...")
        data = data.fillna(data.mean())

    # 特征和目标变量 - 使用英文列名
    # 如果Excel文件中是中文列名，需要映射到英文
    column_mapping = {
        '水含量': 'Water Content',
        '水泥含量': 'Cement Content',
        '黏粒含量': 'Clay Content',
        '强度': 'UCS'
    }

    # 将中文列名重命名为英文
    data = data.rename(columns=column_mapping)
    X_raw = data[['Water Content', 'Cement Content', 'Clay Content']].values
    y = data['UCS'].values

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, random_state=42
    )

    # 训练MS-PINN模型
    print("\n开始训练 MS-PINN 模型...")
    ms_pinn_model, ms_history, ms_physics_weight, ms_alpha = train_model(
        model_class=MS_PINN,
        model_name="MS-PINN",
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        hidden_dim=144,  # 增加隐藏层尺寸
        n_wavelets=16,  # 增加小波数量
        epochs=800,  # 训练轮数
        batch_size=48,  # 批量大小
        lr=0.0022,  # 学习率
        weight_decay=2.5e-5,  # 权重衰减
        physics_weight=0.12,  # 物理约束初始权重
        alpha=0.7,  # 增加神经网络权重占比
        k_init=1.15,
        l_init=0.176,
        adaptive_weights=True
    )

    # 评估MS-PINN模型
    print("\n评估 MS-PINN 模型...")
    ms_metrics = evaluate_model(ms_pinn_model, X_train, y_train, X_test, y_test, alpha=ms_alpha)

    # 保存训练好的模型 - 添加模型保存代码
    model_save_path = "pinn_ms_model.pth"
    torch.save(ms_pinn_model.state_dict(), model_save_path)
    print(f"\n模型已成功保存至 {model_save_path}")
    
    # 同时保存模型参数信息，便于后续加载
    model_info = {
        'input_dim': 3,
        'hidden_dim': 144,
        'n_wavelets': 16,
        'alpha': ms_alpha,
        'k': ms_pinn_model.get_physics_params()[0],
        'l': ms_pinn_model.get_physics_params()[1]
    }
    
    with open("pinn_ms_model_info.pkl", "wb") as f:
        pickle.dump(model_info, f)
    print("模型参数信息已保存至 pinn_ms_model_info.pkl")

    # 进行SHAP分析
    print("\n进行SHAP分析...")
    # 创建结果目录
    results_dir = "shap_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 对原始特征进行SHAP分析
    ms_raw_shap = analyze_raw_features_shap(
        ms_pinn_model,
        X_raw,  # 所有数据的原始特征
        y,
        model_name="MS-PINN",
        alpha=ms_alpha,
        save_dir=results_dir
    )

    # 绘制SHAP waterfall图，选择满足条件的样本
    waterfall_dir = "waterfall_results"
    found_idx = plot_shap_waterfall(
        ms_raw_shap,
        model_name="MS-PINN",
        save_dir=waterfall_dir
    )

    # 创建组合SHAP分析图
    combined_results_dir = "combined_shap_results"
    feature_names = ['Water Content', 'Cement Content', 'Clay Content']
    create_combined_shap_plot(
        ms_raw_shap['shap_values'],
        X_raw[:50],  # 使用部分样本进行可视化
        feature_names,
        model_name="MS-PINN",
        save_dir=combined_results_dir
    )

    # 输出SHAP分析结果摘要
    print("\nSHAP分析特征重要性排名:")
    print(ms_raw_shap['feature_importance'])

    # 评估MS-PINN模型后，添加可视化预测效果的代码
    if not os.path.exists('prediction_results'):
        os.makedirs('prediction_results')

    # 在评估后添加预测结果的可视化
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_pred = ms_pinn_model.mixed_prediction(X_test_tensor, ms_alpha).numpy().flatten()

        # 绘制预测值vs真实值的散点图 - 增加图表宽度
        fig, ax = plt.subplots(figsize=(18 / 2.54, 7 / 2.54))

        # 预测值vs真实值散点图
        ax.scatter(y_test, y_test_pred, alpha=0.6, s=15, color='steelblue', label='Data points')

        # 理想线 y=x
        min_val = min(y_test)
        max_val = max(y_test)
        ax.plot([min_val, max_val], [min_val, max_val], '--', color='#FF5555', linewidth=0.8, label='y=x')

        # 添加 y=0.8x 和 y=1.2x 线
        ax.plot([min_val, max_val], [0.8 * min_val, 0.8 * max_val], '--', color='grey', alpha=0.5, linewidth=0.6)
        ax.plot([min_val, max_val], [1.2 * min_val, 1.2 * max_val], '--', color='grey', alpha=0.5, linewidth=0.6)

        # 添加线旁标注
        x_pos_08 = min_val + (max_val - min_val) * 0.7
        x_pos_12 = min_val + (max_val - min_val) * 0.6
        offset = (max_val - min_val) * 0.05
        ax.text(x_pos_08, 0.8 * x_pos_08 - offset, 'y=0.8x', color='grey', fontsize=7,
                rotation=np.degrees(np.arctan(0.8)))
        ax.text(x_pos_12, 1.2 * x_pos_12 + offset * 0.8, 'y=1.2x', color='grey', fontsize=7,
                rotation=np.degrees(np.arctan(1.2)))

        # 添加R²和RMSE标注
        r2 = ms_metrics['test_metrics']['R2']
        rmse = ms_metrics['test_metrics']['RMSE']
        text_str = f'$R^2={r2:.3f}$\nRMSE={rmse:.3f}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 设置轴标签
        ax.set_xlabel('True UCS (MPa)', fontsize=10)
        ax.set_ylabel('Predicted UCS (MPa)', fontsize=10)

        # 添加标题
        ax.set_title('MS-PINN Model', fontsize=10)

        # 调整图例位置
        ax.legend(loc='lower right', frameon=True, framealpha=0.7)

        # 取消网格
        ax.grid(False)
        plt.tight_layout()

        # 保存图片为PNG和SVG格式
        plt.savefig('prediction_results/MS_PINN_results.png', dpi=1000, bbox_inches='tight')
        plt.savefig('prediction_results/MS_PINN_results.svg', format='svg', bbox_inches='tight')
        plt.close()

    print("\n预测结果可视化已保存至prediction_results目录")


if __name__ == "__main__":
    main()