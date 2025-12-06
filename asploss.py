import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft2, ifft2, fftfreq

class angular_spectrum_loss(nn.Module):
    """
    基于角谱法传播的损失函数，支持mae，mse，rmse
    支持自动维度匹配，输入可以是 (N,N), (B,N,N) 或 (B,1,N,N)
    """
    def __init__(
        self,
        image_size: int,               # 只支持正方形输入
        distance_mm: float = 20.0,
        wavelength_m: float = 632.8e-9,
        pixel_size_m: float = 8e-6,
        pad_factor: float = 1.5,
        loss_type: str = 'mse',
    ):
        super().__init__()
        self.N = image_size
        self.distance_m = distance_mm * 1e-3
        self.wavelength_m = wavelength_m
        self.pixel_size_m = pixel_size_m
        self.loss_type = loss_type
        self.k = 2 * torch.pi / self.wavelength_m

        pad_pixels = int(self.N * (pad_factor - 1) / 2)
        self.pad_pixels = pad_pixels
        self.Np = self.N + 2 * pad_pixels

        f = fftfreq(self.Np, d=self.pixel_size_m)
        fx, fy = torch.meshgrid(f, f, indexing='xy')
        freq_sq = fx**2 + fy**2
        inside = 1 - (self.wavelength_m**2) * freq_sq
        mask = inside > 0
        phase = self.k * self.distance_m * inside.clamp_min(0).sqrt()
        h_filter = torch.exp(1j * phase) * mask
        self.register_buffer('prop', h_filter, persistent=False)
        self.slc = slice(pad_pixels, pad_pixels + self.N)

    def _validate_and_reshape(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        shape = tensor.shape
        ndim = len(shape)
        
        if ndim == 2:  # (N, N) -> (1, 1, N, N)
            return tensor.unsqueeze(0).unsqueeze(0)
        elif ndim == 3:  # (B, N, N) -> (B, 1, N, N)
            return tensor.unsqueeze(1)
        elif ndim == 4:  # (B, 1, N, N)
            if shape[1] != 1:
                raise ValueError(f"{name} 的通道数必须为1，但得到 shape {shape}")
            return tensor
        else:
            raise ValueError(f"{name} 维度不合法，期望2、3或4维，但得到 shape {shape}")

    def forward(self,
                pred_phase: torch.Tensor,
                src_intensity: torch.Tensor,
                target_intensity: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算光强预测与目标之间的损失
    
        参数:
            pred_phase: 预测的相位分布, 支持 shape (N,N), (B,N,N) 或 (B,1,N,N)
            src_intensity: 源平面光强分布, 支持 shape (N,N), (B,N,N) 或 (B,1,N,N)
            target_intensity: 目标平面光强分布（标签）, 支持 shape (N,N), (B,N,N) 或 (B,1,N,N)
    
        返回:
            标量损失值

        注意：
            - 自动处理输入维度，会统一转换为 (B, 1, N, N)
            - 通道维度必须为1，否则会报错
            - 输入需要在同设备上
        """
        pred_phase = self._validate_and_reshape(pred_phase, "pred_phase")
        src_intensity = self._validate_and_reshape(src_intensity, "src_intensity")
        target_intensity = self._validate_and_reshape(target_intensity, "target_intensity")

        if not (pred_phase.shape == src_intensity.shape == target_intensity.shape):
            raise ValueError(
                f"所有输入的形状必须相同，但得到:\n"
                f"  pred_phase: {pred_phase.shape}\n"
                f"  src_intensity: {src_intensity.shape}\n"
                f"  target_intensity: {target_intensity.shape}"
            )

        amplitude = (src_intensity + self.eps).sqrt()
        field = amplitude * torch.exp(1j * pred_phase)
        field = F.pad(field, (self.pad_pixels,)*4)
        field = ifft2(fft2(field, norm='ortho') * self.prop, norm='ortho')[..., self.slc, self.slc]

        output_intensity = field.abs().square()

        if self.loss_type == 'mse':
            return F.mse_loss(output_intensity, target_intensity)
        elif self.loss_type == 'mae':
            return F.l1_loss(output_intensity, target_intensity)
        elif self.loss_type == 'rmse':
            return torch.sqrt(F.mse_loss(output_intensity, target_intensity))
        else:

            raise ValueError(f"不支持损失类型: {self.loss_type}，请选择 'mse', 'mae' 或 'rmse'")
