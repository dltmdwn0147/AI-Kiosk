import pyaudio
import audioop
import sys

# 설정 값
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # 일반적인 오디오 샘플링 레이트

def test_microphone():
    p = pyaudio.PyAudio()

    try:
        # 마이크 스트림 열기
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("\n🎤 [테스트 시작] 마이크에 대고 말씀을 해보세요!")
        print("📊 소리 크기에 따라 막대가 움직입니다. (종료하려면 Ctrl+C)\n")

        while True:
            # 1. 마이크에서 데이터 읽기
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # 2. 소리 크기(RMS) 계산
            rms = audioop.rms(data, 2)  # 2는 sample width (Int16 = 2bytes)
            
            # 3. 소리 크기를 텍스트 막대로 변환 (스케일 조절)
            # rms 값을 적당한 숫자로 나누어 막대 길이를 정합니다.
            level = int(rms / 300) 
            bar = "█" * level
            
            # 4. 터미널에 한 줄로 출력 (이전 줄 지우고 다시 쓰기)
            # ljust(100)은 잔상을 지우기 위해 공백을 채움
            print(f"\rVolume: |{bar.ljust(50)}| (RMS: {rms})", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n🛑 [테스트 종료]")
    except Exception as e:
        print(f"\n❌ [에러 발생] {e}")
        print("👉 Mac 사용자라면 '시스템 설정 > 개인정보 보호 및 보안 > 마이크'에서 터미널(iTerm/VSCode) 권한을 허용했는지 확인하세요.")
    finally:
        # 정리 작업
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    test_microphone()