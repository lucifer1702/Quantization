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

- for unsigned int the range of is 0 to 2 pow(n)-1
- for signed int the range is -2 pow(n) to 2 pow(n)

Floating point numbers are special tho

- they have 3 major stuff
- sign
- exponent
- fraction

For FP32: the sign bit : 1 , the exponent bit is 8 and the fraction bits are 23
the way to compute the number is by :
for sub normal values : (-1) pow(s) _F2 pow(-126)
and for normal values : (-1) pow(s) _(1+F)\*2 pow(E-127)

same way to calculate for bf16 but the way is different

- bf16 : 1 sign bit , 7 exponent bits , and 8 fraction bits

for float 16 : 1 sign bit , 5 exponent bits and 10 fraction bits

- Downcasting : process of casting from higher precison to lower precision

### Linear Quantization

- In linear quantization we basically try to go from one precision to the other whilst minimizing the loss
- In this case for instance : we map the maximum values of the tensor to maximum value of the least precise dtype
- we also have a size and zero mapping which will help us dequantize it
- in the case of linear quantization this is used in many sota techniques such as awq(activcation aware weight quantization), gptq(gpt quantization) and BnB(bits and bytes )

- Now some more pretty interesting stuff :
  lets say you want to go from dtype original -> to dtype quantized
  how to do
  basically r = s(q-z)
  where s: original dtype
  q: quantized dtype
  s: scale factor
  z: zero point factor

- now to calc s: pretty simple as you guessed
  s = (rmax-rmin)/(qmax-qmin)
  z = int(round(qmin-rmin/s))

now remeber when you used the quanto library there is an intermediate step before you used freezing
that turns out to be quite helpful as it contains both the quantized and original activation teensors of the model
hence we can use that on two things

1. calibration
   whilst inference we can see how the min and max values of the activation tensors are and can calibrate them
2. quantization aware training
   since the int state has both quantized and original activation tensors we can use them whilst training

- basically during the forward pass we train the quantized weights : basically inference
- whilst back prop we train the original weights : training time
