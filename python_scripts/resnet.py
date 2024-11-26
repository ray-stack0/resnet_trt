import torchvision
import torch
import struct

def save_wts(model:torch.nn.Module,model_name):
    f = open(f"{model_name}.wts", 'w')
    f.write(f"{len(model.state_dict().keys())}\n")
    for name, params in model.state_dict().items():
        # print(f'key;{name}')
        # print(f'value:{params.shape}')
        params = params.reshape(-1).cpu().numpy()
        f.write(f'{name} {len(params)}')
        for param in params:
            f.write(' ')
            f.write(struct.pack(">f",float(param)).hex())
        f.write('\n')


input = torch.randn(1,3,224,224)
resnet50 = torchvision.models.resnet50(weights='IMAGENET1K_V2')
resnet50.eval()

with torch.no_grad():
    torch.onnx.export(
        resnet50,
        input,
        "resnet50.onnx",
        input_names=['input_image'],
        output_names=['cls_pro'],
        opset_version=11,
        export_params=True,
        do_constant_folding=False
    )
    save_wts(resnet50,'resnet50')
