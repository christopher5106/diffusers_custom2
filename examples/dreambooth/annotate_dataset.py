from PIL import Image
from pathlib import Path

# from transformers import BlipProcessor, BlipForConditionalGeneration
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

import torch
from lavis.models import load_model_and_preprocess

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="pretrain_flant5xxl",
    is_eval=True,
    device=device)

datasets = Path("/home/ubuntu/datasets/")

for f in [datasets / "golem", datasets / "vendalixia"]:  # datasets.iterdir():
    if f.is_dir():
        for image_path in f.iterdir():
            print(image_path)
            if str(image_path)[-4:] in [".png", ".jpg", ".jpeg"]:
                raw_image = Image.open(image_path).convert('RGB')
                image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

                # inputs = processor(raw_image, return_tensors="pt").to("cuda")
                # out = model.generate(**inputs)
                # caption = processor.decode(out[0], skip_special_tokens=True)
                caption = model.generate({"image": image})
                # caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
                print("   ", caption)
                with open(str(image_path)+".txt", "w") as _f:
                    _f.write(caption)
