import asyncio
import os
import pyaudio
import sys
from dotenv import load_dotenv
from google import genai

# 1. API 키 로드
load_dotenv()
MY_API_KEY = os.getenv("GEMINI_API_KEY")

if not MY_API_KEY:
    raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 2. 오디오 설정 (Gemini Live API 권장 규격)
# 주의: 일반적인 44100Hz 대신 16000Hz를 사용해야 데이터 전송량이 줄고 인식이 빠릅니다.
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 
CHUNK = 1024

# 3. 모델 및 클라이언트 설정
client = genai.Client(api_key=MY_API_KEY, http_options={'api_version': 'v1alpha'})
MODEL_ID = "gemini-2.0-flash-exp"  # 범용 모델 (안정적)
CONFIG = {"response_modalities": ["TEXT"]} # 응답은 텍스트로만 받음

class AudioHandler:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None

    def start_stream(self):
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        print("\n🎤 [연결 완료] 이제 말씀하세요! Gemini가 듣고 있습니다...")
        print("   (종료하려면 키보드에서 Ctrl+C 를 누르세요)\n")

    def read_chunk(self):
        if self.stream and self.stream.is_active():
            return self.stream.read(CHUNK, exception_on_overflow=False)
        return None

    def close(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("\n🛑 [오디오 장치 종료]")

async def send_audio_loop(session, audio_handler):
    """마이크 소리를 끊임없이 서버로 전송"""
    while True:
        try:
            data = audio_handler.read_chunk()
            if data:
                # Gemini에게 바이너리 오디오 데이터 전송
                await session.send(input={"data": data, "mime_type": "audio/pcm"}, end_of_turn=False)
            await asyncio.sleep(0.001) # 루프 과부하 방지
        except Exception as e:
            print(f"⚠️ 전송 중 오류: {e}")
            break

async def receive_response_loop(session):
    """서버에서 오는 텍스트 응답을 실시간으로 출력"""
    try:
        async for response in session.receive():
            if response.text:
                # 텍스트가 조각(chunk)으로 들어오므로 줄바꿈 없이 이어서 출력
                print(f"{response.text}", end="", flush=True)
                
                # 문장이 끝난 것 같으면 줄바꿈 (선택사항)
                if response.text.endswith((".", "?", "!")):
                    print() 
    except Exception as e:
        print(f"\n⚠️ 수신 중 오류: {e}")

async def main():
    audio_handler = AudioHandler()
    
    try:
        # Live API 세션 시작
        async with client.aio.live.connect(model=MODEL_ID, config=CONFIG) as session:
            
            # 마이크 켜기
            audio_handler.start_stream()

            # '보내기'와 '받기'를 동시에 실행 (비동기 병렬 처리)
            send_task = asyncio.create_task(send_audio_loop(session, audio_handler))
            receive_task = asyncio.create_task(receive_response_loop(session))

            # Ctrl+C가 눌리기 전까지 무한 대기
            await asyncio.gather(send_task, receive_task)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        # 여기가 중요: 아까 발생했던 Quota(한도) 에러가 나면 여기서 잡힙니다.
        if "429" in str(e) or "Quota" in str(e):
            print("\n❌ [한도 초과] 무료 사용량을 모두 소진했습니다. 유료/대회용 계정이 필요합니다.")
        else:
            print(f"\n❌ 연결 오류 발생: {e}")
    finally:
        audio_handler.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 프로그램을 종료합니다.")