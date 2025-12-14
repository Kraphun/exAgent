import torch
import os
import json
from dataclasses import dataclass
from typing import Dict, List, Union, Any

from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
from qwen_vl_utils import process_vision_info

# Hyper parameters
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DATA_FILE = "./Degradation/train_data_augmented.json"
OUTPUT_DIR = "./checkpoints/qlora/qwen2-vl-agent-checkpoint"

# 3070 Ti (8GB) 맞춤 설정
MAX_SEQ_LENGTH = 1024  # 이미지 크기 - 텍스트 길이 반비례하게 조절 필요
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Custom Data Collator
@dataclass
class Qwen2VLDataCollator:
    """
    Qwen2-VL은 텍스트와 이미지를 동시에 처리해서 
    input_ids, pixel_values, image_grid_thw 등을 만들어야 합니다.
    이를 배치 단위로 묶어주는 역할을 합니다.
    """
    processor: AutoProcessor

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. 텍스트와 이미지 데이터 추출
        texts = []
        image_inputs = []
        
        for example in examples:
            raw_messages = example["messages"]

            # role에 따라 메세지 정리 -> 정리 안하면, assistant role까지 이미지로 처리해서 오류(line 64) 생겼음
            messages = []
            for msg in raw_messages:
                role = msg["role"]
                content = msg["content"]

                clean_content = []
                if isinstance(content, list):
                    for item in content:
                        clean_item = {k: v for k, v in item.items() if v is not None}
                        clean_content.append(clean_item)
                else:
                    clean_content = content

                messages.append({"role": role, "content": clean_content})

            # 프롬프트 처리 (Qwen-VL Utils 활용)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            
            # 이미지 처리
            image_input, video_input = process_vision_info(messages)
            image_inputs.append(image_input)

        # 2. Processor를 통해 텐서 변환 (Padding 적용)
        batch = self.processor(
            text=texts,
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 3. Label 생성
        # 기본적으로 input_ids를 복사, 패딩 토큰은 -100으로 마스킹해서 Loss 계산 제외
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        batch["labels"] = labels
        return batch

def train():
    print(f">>> 학습 준비 시작 (Device: {torch.cuda.get_device_name(0)})")
    
    # A. 4-bit 양자화 설정 (QLoRA) - 3070ti를 위해 ㅜㅜ
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # B. 모델 및 프로세서 로드
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        # attn_implementation="flash_attention_2" # todo: flash attention적용 필요.. 윈도우에서 ㅠㅠ
    )
    
    # VRAM 넉넉한 곳에서는 아래 옵션 없이 테스트 해보기
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # C. LoRA 어댑터 설정
    # Frozen : Vision Encoder
    # Tunable : LLM part
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    
    
    # load dataset
    print(f">>> 데이터셋 로드: {DATA_FILE}")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    # Set trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,      
        gradient_accumulation_steps=16,     
        num_train_epochs=3,                 # 데이터가 적으므로 3~5 epoch
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=True,                          
        optim="paged_adamw_8bit",           # 옵티마이저 메모리 절약
        remove_unused_columns=False,        # VLM 데이터셋 컬럼 유지
        report_to="none",                   
        dataloader_pin_memory=False,        # 윈도우에서 가끔 오류 발생 방지
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=Qwen2VLDataCollator(processor),
    )

    # Do training
    print(">>> 학습 시작! (시간이 좀 걸립니다...)")
    trainer.train()
    
    # Save final model
    print(f">>> 학습 완료! 모델 저장 중: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()