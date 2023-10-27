#import onnx
#from onnxruntime.quantization import quantize_dynamic, QuantType
#
#model_fp32 = 'model.onnx'
#model_quant = 'model.quant.onnx'
#quantized_model = quantize_dynamic(model_fp32, model_quant)

from onnxruntime.quantization import QuantType, quantize_dynamic
 
# 模型路径
model_fp32 = 'model.onnx'
model_quant_dynamic = 'MobileNetV1_infer_quant_dynamic.onnx'
 
# 动态量化
quantize_dynamic(
    model_input=model_fp32, # 输入模型
    model_output=model_quant_dynamic, # 输出模型
    weight_type=QuantType.QUInt8, # 参数类型 Int8 / UInt8
)