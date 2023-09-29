import torch
from SuperModel import SuperResolutionModel
from torchvision import transforms
from PIL import Image

if __name__ == "__main__":

    # 创建与训练时相同架构的模型实例
    model = SuperResolutionModel()

    # 加载模型权重
    model.load_state_dict(torch.load("SuperResolution/super_resolution_model.pth"))

    # 加载测试图像
    test_image = Image.open("SuperResolution/testdata/input/1.jpg")

    # 将测试图像转换为模型输入所需的格式（例如，将其调整为模型期望的大小）
    # 这里假设模型期望的输入大小是256x256
    test_image = test_image.resize((100, 100))
    test_image_tensor = transforms.ToTensor()(test_image).unsqueeze(0)  # 添加批量维度

    # 使用模型进行超分辨率处理
    with torch.no_grad():
        super_res_image = model(test_image_tensor)

    # 将模型输出转换为PIL图像
    super_res_image_pil = transforms.ToPILImage()(super_res_image.squeeze(0))

    # 保存高分辨率图像
    super_res_image_pil.save("SuperResolution/testdata/output/1_output.jpg")
