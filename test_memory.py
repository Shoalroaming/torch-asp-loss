import torch
from asploss import angular_spectrum_loss

def mem():
    return torch.cuda.memory_allocated() / 1024**2

def peak():
    return torch.cuda.max_memory_allocated() / 1024**2

def test_single():
    print("\n--- 单尺寸测试 ---")
    torch.cuda.empty_cache()
    
    loss_fn = angular_spectrum_loss(image_size=256).cuda()
    pred = torch.randn(4, 256, 256, device='cuda', requires_grad=True)
    src = target = torch.ones(4, 256, 256, device='cuda')
    
    init = mem()
    loss = loss_fn(pred, src, target)
    loss.backward()
    
    print(f"初始: {init:.1f} MB → 峰值: {peak():.1f} MB → 最终: {mem():.1f} MB")

def test_multi():
    print("\n--- 多尺寸测试 ---")
    torch.cuda.empty_cache()
    
    loss_fns = {s: angular_spectrum_loss(image_size=s).cuda() for s in [256, 512, 1024]}
    init = mem()

    for s in [256, 512, 1024]:
        pred = torch.randn(2, s, s, device='cuda', requires_grad=True)
        src = target = torch.ones(2, s, s, device='cuda')
        
        loss = loss_fns[s](pred, src, target)
        loss.backward()
    
    print(f"初始: {init:.1f} MB → 峰值: {peak():.1f} MB → 最终: {mem():.1f} MB")

    del pred, src, target, loss
    torch.cuda.empty_cache()

def main():
    test_single()
    test_multi()

if __name__ == "__main__":
    main()