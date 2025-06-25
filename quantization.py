import torch
from transformers import (
    AutoModelForCasualLM,
    AutoTokenizer,
    BlipForConditionalGenration,
)

model = BlipForConditionalGenration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
# to get the memory foot print of the model
print(model.get_memory_footprint())
# to convert it into lets say into bf16
model_bf16 = BlipForConditionalGenration.from_pretrained(
    "model_name", torch_dtype=torch.bfloat16
)

# both are image captioning models loaded in different precision hence the logits generated during the eval will also be different

# instead of loading first in full precision and then casting it into lets say bf16
# we can load directly in the desired format
type_float = torch.bfloat16

torch.set_default_dtype(type_float)

# quanto
model_name = "Eleuther/pythia-410m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCasualLM.from_pretrained(
    model_name, low_cpu_mem_usage=True
)  # the low cpu mem usage flag is for the cpu usage to be optimized
inputs = tokenizer(
    text, return_tensors="pt"
)  # the return tensors_param = "pt"  basicaly is for saying the return tensors should be pytorhc tensors
outputs = model.generate(
    **inputs
)  # ** is used as the inputs are dictinoary of arguments
# to decode
use
tokenizer.decode(outputs[0], skip_special_tokens=True)
