This is the repo for everything about learning quantization

1. Basics

- used in shrinking models
- there is growing gap in the size of the model and the compute available
- as models get larger it becomes increasingly difficult to perform computations
- fpr instance a 7b param model needs approimately 280gb of compute
- consumer grade T4 gPUS only have 16GB of memory

2. Some of the commonly used techniques to help with this are

- Pruning ; which basically prunes weight and nodes based on some condition
- Knowledge distillation - where in You use a student model which is trained on the parent model's output and a loss function
  but these methods are also computationly expensive hence the need to pivot to quantization

## Quantization

The process of converting data into lower precision data types

- common dtypes used are FP32,FP16,BF16 and INT8
- the whole process of Quantization is to reduce the quantization error that comes with this conversion
