import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from asploss import angular_spectrum_loss
from skimage.restoration import unwrap_phase
from asp import angular_spectrum_propagation

def target(size, pattern='circle'):
    x, y = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='xy')
    if pattern == 'circle':
        return ((x**2 + y**2) < 0.3**2).float().unsqueeze(0).unsqueeze(0)
    elif pattern == 'letter_x':
        return ((abs(x) + abs(y)) < 0.5).float().unsqueeze(0).unsqueeze(0)
    else:
        return torch.rand(1, 1, size, size)

def visualize(pred_phase, target_intensity, output_intensity, losses):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    phase_np = pred_phase.detach().cpu().squeeze().numpy()
    phase_np = unwrap_phase(phase_np)
    target_np = target_intensity.cpu().squeeze().numpy()
    output_np = output_intensity.detach().cpu().squeeze().numpy()

    vmax = max(target_np.max(), output_np.max())

    axes[0, 0].imshow(target_np, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 0].set_title('目标灰度图')

    axes[0, 1].imshow(output_np, cmap='gray', vmin=0, vmax=vmax)
    axes[0, 1].set_title('输出灰度图')

    im1 = axes[1, 0].imshow(phase_np, cmap='twilight')
    axes[1, 0].set_title('模拟相位图')
    divider1 = make_axes_locatable(axes[1, 0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im1, cax=cax1)

    axes[1, 1].plot(losses)
    axes[1, 1].set_title('损失曲线')
    axes[1, 1].set_yscale('log')
    axes[1, 1].yaxis.set_major_formatter(ScalarFormatter(useMathText=False))

    plt.tight_layout()
    plt.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image_size = 256
    loss_fn = angular_spectrum_loss(
        image_size=image_size,
        distance_mm=20.0,
        loss_type='mse'
    ).to(device)

    src_intensity = torch.ones(1, 1, image_size, image_size, device=device)
    target_intensity = target(image_size, pattern='circle').to(device)
    pred_phase = torch.randn(1, 1, image_size, image_size, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([pred_phase], lr=0.1)

    losses = []
    for step in range(100):
        optimizer.zero_grad()
        loss = loss_fn(pred_phase.squeeze(), src_intensity.squeeze(), target_intensity.squeeze())
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    with torch.no_grad():
        final_intensity = angular_spectrum_propagation(
            phase=pred_phase.squeeze(),
            intensity=src_intensity.squeeze(),
            distance_mm=getattr(loss_fn,'distance_mm', 20.0),
            wavelength_m=getattr(loss_fn, 'wavelength_m', 632.8e-9),
            pixel_size_m=getattr(loss_fn, 'pixel_size_m', 8e-6),
            pad_factor=getattr(loss_fn, 'pad_factor', 1.5)
        ).unsqueeze(0).unsqueeze(0)

    visualize(pred_phase, target_intensity, final_intensity, losses)
    print(f"最终损失: {losses[-1]:.4e}")

if __name__ == '__main__':
    main()