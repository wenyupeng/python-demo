# 定义模块的公开接口
__all__ = ["train_step", "val_step", "create_embedding"]

# 导入PyTorch核心库和神经网络模块
import torch

def train_step(encoder, decoder, train_loader, loss_fn, optimizer, device):
    """
    执行一个完整的训练迭代

    参数:
    - encoder: 卷积编码器（如ConvEncoder）
    - decoder: 卷积解码器（如ConvDecoder）
    - train_loader: 训练数据加载器，提供批次化的（输入图像, 目标图像）
    - loss_fn: 损失函数（如MSE）
    - optimizer: 优化器（如Adam）
    - device: 计算设备（"cuda" 或 "cpu"）

    返回值:
    - 当前epoch的平均训练损失（标量值）
    """

    total_loss = 0  # 累计损失
    num_batches = 0  # 批次计数器

    # 遍历训练数据加载器中的所有批次
    for train_img, target_img in train_loader:
        # 将数据移动到指定设备（GPU/CPU）
        train_img = train_img.to(device)
        target_img = target_img.to(device)

        # 清空优化器中之前的梯度
        optimizer.zero_grad()
        # 前向传播：编码器生成潜在表示
        enc_output = encoder(train_img)
        # 前向传播：解码器重建图像
        dec_output = decoder(enc_output)

        # 计算重建损失（预测图像与目标图像的差异）
        loss = loss_fn(dec_output, target_img)

        # 反向传播：计算梯度
        loss.backward()

        # 优化器更新模型参数
        optimizer.step()

        total_loss += loss.item()  # 累加损失值
        num_batches += 1
        # 返回平均损失
    return total_loss / num_batches

def val_step(encoder, decoder, val_loader, loss_fn, device):
    """
    执行验证步骤（不更新参数）

    参数与train_step类似，但不含优化器参数

    返回值:
    - 验证集的平均损失（标量值）
    """

    total_loss = 0
    num_batches = 0

    # 禁用梯度计算以节省内存和计算资源
    with torch.no_grad():
        for train_img, target_img in val_loader:
            train_img = train_img.to(device)
            target_img = target_img.to(device)

            # 前向传播
            enc_output = encoder(train_img)
            dec_output = decoder(enc_output)

            # 计算损失
            loss = loss_fn(dec_output, target_img)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def create_embedding(encoder, full_loader, embedding_dim, device):
    """
    使用编码器为整个数据集生成嵌入表示

    参数:
    - encoder: 训练好的编码器
    - full_loader: 完整数据集的数据加载器
    - embedding_dim: 预期嵌入的维度（元组格式 (c, h, w)）
    - device: 计算设备

    返回值:
    - 嵌入张量，形状为 (num_samples + 1, c, h, w)
    """

    # 初始化随机嵌入（注：此处可能应为空张量，初始随机值可能引入噪声）
    # embedding = torch.randn(embedding_dim)  # 初始随机张量
    embedding = torch.empty(0) # 初始为空张量

    with torch.no_grad():
        for train_img, target_img in full_loader:
            # 仅使用输入图像（自编码器结构）
            train_img = train_img.to(device)

            # 生成当前批次的嵌入
            enc_output = encoder(train_img).cpu()  # 将结果移回CPU

            # 沿批次维度拼接嵌入（可能导致内存问题，建议预分配空间）
            embedding = torch.cat((embedding, enc_output), 0)

    return embedding  # 返回所有样本的嵌入集合
