import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from PIL import Image

def main():
    model_name = "NCSOFT/VARCO-VISION-2.0-1.7B"

    # 모델 로드 (Jetson Orin: FP16 + device_map="auto")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto",
    ).eval()

    processor = AutoProcessor.from_pretrained(model_name)

    # 테스트할 이미지 경로 (원하는 이미지로 교체)
    image_path = "a.jpeg"
    image = Image.open(image_path).convert("RGB")

    # 간단한 한국어 프롬프트 (상황 설명)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "복도에 사람이 몇 명 있는지"},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, torch.float16)

    with torch.inference_mode():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,   # caption은 deterministic하게
        )

    # 프롬프트 부분 잘라내기
    generate_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generate_ids)
    ]
    output_text = processor.decode(generate_ids_trimmed[0], skip_special_tokens=True)
    print("=== 모델 출력 ===")
    print(output_text)

if __name__ == "__main__":
    main()
