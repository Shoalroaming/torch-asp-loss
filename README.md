# 角谱法衍射传播损失

基于PyTorch的角谱法衍射传播损失函数，支持GPU加速。

## 功能特性

- 角谱法衍射传播
- 多种损失类型
- 自动维度处理
- GPU加速
- 内存优化

## 文件结构

```
.
├── asp.py              # 角谱法
├── asploss.py          # 角谱法损失
├── test_memory.py      # 内存使用测试
└── test_optimize.py    # 相位优化测试
```

## 核心类

### angular_spectrum_loss

```python
angular_spectrum_loss(
    image_size,         # 图像尺寸（正方形）
    distance_mm=20.0,   # 传播距离 [mm]
    wavelength_m=632.8e-9, # 波长 [m]
    pixel_size_m=8e-6,  # 像素尺寸 [m]
    pad_factor=1.5,     # 填充因子（防止混叠）
    loss_type='mse',    # 损失类型：'mse', 'mae', 'rmse'
    eps=1e-8            # 数值稳定性常数
)
```

## 使用示例

### 内存测试

```bash
python test_memory.py
```

测试不同图像尺寸下的内存使用情况，验证GPU内存效率。

### 相位优化测试

```bash
python test_optimize.py
```

使用梯度下降优化随机相位，使其传播后匹配目标光强分布，包含可视化结果。

## 输入输出

### 输入：

- 相位图 + 光强图 + 目标光强图（维度为(H,W) 或 (B,H,W) 或（B，1，H，W））
- 波长、像素尺寸等物理参数
### 输出：
- 传播后光强与目标光强的损失值


## 依赖

- PyTorch
- NumPy（仅可视化脚本）
- matplotlib (仅可视化脚本)
- scikit-image (仅可视化脚本)
