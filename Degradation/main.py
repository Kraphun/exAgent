import torch
import os

from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------
# 1. State 정의 (데이터 DTO 역할)
# ---------------------------------------------------------
class AgentState(TypedDict):
    image_path: str
    analysis_result: Optional[str]
    final_report: Optional[str]

# ---------------------------------------------------------
# 2. AI Service Class (모델 관리 및 추론 담당)
# ---------------------------------------------------------
class ImageAnalysisService:
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cuda"):
        """
        서비스 초기화 시 모델을 로드하여 메모리에 상주깁니다.
        (API 서버 시작 시 1회만 호출됨)
        """
        self.device = device
        self.model_id = model_id
        self.model = None
        self.processor = None
        
        print(f"[{self.__class__.__name__}] 모델 초기화 시작... Target Device: {self.device}")
        self._load_model()
        print(f"[{self.__class__.__name__}] 모델 로드 완료! 서비스 준비됨.")

    def _load_model(self):
        """내부 메서드: 실제 모델 로딩 로직"""
        #todo: flash attention 적용, window에서..
        try:
            # RTX 3070 Ti (Ampere) -> bfloat16 지원 + Flash Attention 권장
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="flash_attention_2"
            )
        except Exception as e:
            print(f"Warning: Flash Attention 2 로드 실패. 호환 모드로 전환합니다. Error: {e}")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                device_map=self.device
            )
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def analyze_image(self, image_path: str) -> str:
        """
        실제 추론(Inference)을 수행하는 메서드
        """
        prompt = (
            "Analyze this image technically. "
            "Does this image have any quality degradation? "
            "Check for Blur, Gaussian Noise, JPEG Compression artifacts, or Low Resolution. "
            "Answer in this format:\n"
            "- Degradation Detected: [Yes/No]\n"
            "- Type: [Type or None]\n"
            "- Severity: [Low/Medium/High]\n"
            "- Description: [Brief explanation]"
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 전처리
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # 추론
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        # 후처리 (Decoding)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return output_text

# ---------------------------------------------------------
# 3. LangGraph Workflow 정의 (비즈니스 로직)
# ---------------------------------------------------------
def create_workflow(service_instance: ImageAnalysisService):
    """
    초기화된 Service Instance를 주입받아 Graph를 생성합니다.
    """
    
    # Node 1: 모델 추론 호출
    def detect_degradation(state: AgentState):
        image_path = state["image_path"]
        print(f"\n[Workflow] 이미지 분석 요청: {image_path}")
        result = service_instance.analyze_image(image_path)
        return {"analysis_result": result}

    # Node 2: 리포트 포맷팅
    def format_report(state: AgentState):
        result = state["analysis_result"]
        report = f"--- [AI Master Report] ---\n{result}\n--------------------------"
        return {"final_report": report}

    # 그래프 구성
    workflow = StateGraph(AgentState)
    workflow.add_node("detect_degradation", detect_degradation)
    workflow.add_node("generate_report", format_report)

    workflow.set_entry_point("detect_degradation")
    workflow.add_edge("detect_degradation", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()

# ---------------------------------------------------------
# 4. Main Execution (추후 API 서버의 lifespan과 유사)
# ---------------------------------------------------------
if __name__ == "__main__":
    # [Step 1] 서버 시작 시점 (Startup Event)
    # 서비스를 인스턴스화 합니다. 이때 모델이 GPU에 로드됩니다.
    # 이 인스턴스는 프로그램이 종료될 때까지 메모리에 유지됩니다.
    ai_service = ImageAnalysisService()
    
    # [Step 2] 워크플로우 구성
    # 서비스 인스턴스를 그래프에 주입(Dependency Injection)합니다.
    app = create_workflow(ai_service)

    # [Step 3] 요청 처리 (Request Handling)
    test_image_path = "dataset/Rain100L/rainy/rain-001.png" 

    # 테스트 이미지 생성
    if not os.path.exists(test_image_path):
        import numpy as np
        im_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(im_array).save(test_image_path)

    print("\n>>> [Request] 사용자 요청 도착")
    inputs = {"image_path": test_image_path}
    final_result = app.invoke(inputs)
    
    print("\n>>> [Response] 최종 응답 전송:")
    print(final_result["final_report"])


    
    # test_image_path = "dataset/Rain100L/rainy/rain-001.png" 
