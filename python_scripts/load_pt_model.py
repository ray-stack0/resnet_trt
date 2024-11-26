import torch


input = torch.randn(1,1,148)
model = torch.jit.load('cruise_go_vehicle_model.pt')
model.eval()
with torch.no_grad():
    torch.onnx.export(
        model,
        input,
        "resnet50.onnx",
        input_names=['input_image'],
        output_names=['cls_pro'],
        opset_version=11,
        export_params=True
    )
