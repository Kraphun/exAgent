# exAgent

sVLM Qwen VL 2B 기반 Agent 연습
1. sVLM 기반 훼손 판단 에이전트 생성
2. QLora 기반 sVLM Instrunction Tuning (Post Training)

메인 함수 : main.py
Streamlit 실행 : streamlit run .\Degradation\app.py --server.address 127.0.0.1

history
1. baseline 코드 개발
2. Qlora 학습용 데이터셋 생성 코드 개발
3. Qlora 학습 코드 개발
4. Qlora 학습 완료
5. Qlora 적용 -> 메인, 웹페이지 코드 업데이트