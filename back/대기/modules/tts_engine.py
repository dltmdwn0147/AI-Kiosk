import edge_tts
import pygame
import asyncio
import io  # [추가] 메모리 처리를 위한 라이브러리
import time

class TTSEngine:
    def __init__(self, voice="ko-KR-SunHiNeural", rate="+0%", volume="+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        
        try:
            pygame.mixer.init()
            print("[TTS Engine] 오디오 장치 초기화 완료")
        except Exception as e:
            print(f"[TTS Engine] 오디오 장치 초기화 실패: {e}")

    async def _fetch_audio_data(self, text):
        """
        (내부 함수) 텍스트 -> 오디오 데이터(Bytes) 변환
        파일로 저장하지 않고, 메모리 상의 데이터 덩어리(bytes)를 반환합니다.
        """
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, volume=self.volume)
        
        # 스트리밍된 데이터를 모을 바이트 배열
        audio_data = b""
        
        # stream()을 사용하여 청크 단위로 데이터를 받습니다.
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
                
        return audio_data

    def _play_from_memory(self, audio_bytes):
        """(내부 함수) 메모리에 있는 오디오 데이터 재생"""
        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.stop()
                pygame.mixer.music.unload() # 기존 데이터 비우기

            # 바이트 데이터를 마치 파일인 것처럼 포장(Wrapping)합니다.
            sound_file = io.BytesIO(audio_bytes)
            
            # 파일 경로 대신 메모리 객체를 로드합니다.
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                
        except Exception as e:
            print(f"[TTS Engine] 재생 오류: {e}")

    def speak(self, text):
        if not text: return

        print(f"🗣️ [TTS 출력]: {text}")
        
        try:
            # 1. 파일 생성이 아니라, 데이터(bytes)를 직접 받아옵니다.
            audio_bytes = asyncio.run(self._fetch_audio_data(text))
            
            # 2. 받아온 데이터를 바로 재생합니다.
            self._play_from_memory(audio_bytes)
            
        except Exception as e:
            print(f"[TTS Engine] 처리 실패: {e}")

if __name__ == "__main__":
    tts = TTSEngine()
    tts.speak("파일을 저장하지 않고 메모리에서 바로 재생합니다.")
    tts.speak("속도가 더 빠르고 디스크를 쓰지 않아 효율적입니다.")