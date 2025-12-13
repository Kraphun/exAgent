import streamlit as st
import os

# ---------------------------------------------------------
# 1. 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Image Inspector",
    page_icon="🔍",
    layout="wide"
)
print(">>> start Degradation app page")

from PIL import Image
import shutil

from main import ImageAnalysisService, create_workflow

st.title("🔍 AI Image Degradation Inspector")
st.markdown("""
**Qwen2-VL 기반 이미지 훼손 분석 에이전트**입니다.  
이미지를 업로드하면 훼손 여부(Blur, Noise 등)를 판단하고 리포트를 생성합니다.
""")

# ---------------------------------------------------------
# 2. 모델 로드 (캐싱 적용)
@st.cache_resource
def get_ai_service():
    """
    서비스 인스턴스를 캐싱합니다.
    이 함수는 앱이 실행되는 동안 최초 1회만 실행되며,
    이후에는 이미 로드된 인스턴스를 반환합니다.
    """
    with st.spinner("AI 모델을 GPU(RTX 3070 Ti)에 로드 중입니다... 잠시만 기다려주세요."):
        # 3070 Ti 메모리 최적화를 위해 로드
        service = ImageAnalysisService()
        return service

with st.sidebar:
    st.header("System Status")
    try:
        ai_service = get_ai_service()
        st.success("✅ Model Loaded (Warm State)")
        st.info(f"Device: {ai_service.device}")
    except Exception as e:
        st.error(f"❌ Model Load Failed: {e}")
        st.stop()

workflow_app = create_workflow(ai_service)


col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. 이미지 업로드")
    uploaded_file = st.file_uploader("분석할 이미지를 선택하세요", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # 업로드된 파일을 임시 경로에 저장 (백엔드가 경로를 요구하므로)
        # 실무에서는 tempfile 라이브러리를 쓰거나 S3에 올립니다.
        os.makedirs("temp", exist_ok=True)
        temp_path = os.path.join("temp", uploaded_file.name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 이미지 미리보기
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    st.subheader("2. 분석 결과")
    
    if uploaded_file is not None:
        analyze_btn = st.button("🚀 이미지 분석 시작", type="primary")
        
        if analyze_btn:
            with st.spinner("AI가 이미지를 분석하고 있습니다..."):
                try:
                    # LangGraph 워크플로우 실행
                    inputs = {"image_path": temp_path}
                    result = workflow_app.invoke(inputs)
                    
                    final_report = result.get("final_report", "No result generated.")
                    
                    # 결과 출력
                    st.success("분석 완료!")
                    st.text_area("Analysis Report", value=final_report, height=300)
                    
                    # 추가적인 시각적 피드백 (예시)
                    if "Degradation Detected: Yes" in final_report:
                        st.warning("⚠️ 이미지 훼손이 감지되었습니다.")
                    else:
                        st.balloons()
                        st.info("✅ 이미지가 깨끗한 것으로 보입니다.")
                        
                except Exception as e:
                    st.error(f"분석 중 오류 발생: {e}")
                    
    else:
        st.info("왼쪽에서 이미지를 먼저 업로드해주세요.")

# ---------------------------------------------------------
# 5. 리소스 정리 (Optional)
# ---------------------------------------------------------
# 실제 배포 시에는 주기적으로 temp 폴더를 비워주는 로직이 필요합니다.