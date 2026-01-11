"""
Mediapipe FaceLandmarker 모델 파일 다운로드 스크립트
"""
import os
import urllib.request
from pathlib import Path

def download_face_landmarker_model(model_dir=None):
    """
    Mediapipe FaceLandmarker 모델 파일을 다운로드합니다.
    
    Args:
        model_dir: 모델 파일을 저장할 디렉토리 (기본값: 현재 스크립트 디렉토리)
    
    Returns:
        모델 파일 경로
    """
    if model_dir is None:
        model_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_file = model_dir / "face_landmarker.task"
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    if model_file.exists():
        print(f"✅ 모델 파일이 이미 존재합니다: {model_file}")
        return str(model_file)
    
    print(f"📥 모델 파일 다운로드 중: {model_url}")
    print(f"📁 저장 위치: {model_file}")
    
    try:
        # 다운로드 진행률 표시
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"\r⏳ 다운로드 진행률: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='', flush=True)
        
        urllib.request.urlretrieve(model_url, model_file, show_progress)
        print(f"\n✅ 모델 파일 다운로드 완료: {model_file}")
        print(f"📊 파일 크기: {model_file.stat().st_size / (1024*1024):.2f} MB")
        return str(model_file)
    except Exception as e:
        print(f"\n❌ 모델 파일 다운로드 실패: {e}")
        if model_file.exists():
            model_file.unlink()  # 실패한 파일 삭제
        raise

if __name__ == '__main__':
    try:
        model_path = download_face_landmarker_model()
        print(f"\n✅ 모델 파일 준비 완료: {model_path}")
        print("💡 이제 main.py를 실행하면 안면 인식을 사용할 수 있습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💡 인터넷 연결을 확인하고 다시 시도해주세요.")

