import torch
from transformers import BlipForConditionalGenration

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
