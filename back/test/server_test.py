import socket
import time
import sys

# 실행 확인용 메시지 (이게 안 뜨면 파일이 비어있는 것임)
print("--- server_test.py 파일을 읽기 시작했습니다 ---")

HOST = '127.0.0.1'
PORT = 9999

def run_server():
    print("1. 소켓 생성 중...")
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # 포트 재사용 옵션
        
        print(f"2. 바인딩 시도 ({HOST}:{PORT})...")
        server_socket.bind((HOST, PORT))
        
        server_socket.listen()
        print(f"🚀 [Server] Listening on {HOST}:{PORT}")
        print("⏳ 키오스크(Client)의 접속을 기다리는 중입니다... (터미널 끄지 마세요)")

        # 여기서 대기함 (키오스크가 들어올 때까지 멈춰있어야 정상)
        client_soc, addr = server_socket.accept()
        print(f"✅ [Server] Client connected: {addr}")

        # 3초 뒤에 '아메리카노' 주문 전송
        print("📤 3초 뒤에 '아메리카노' 주문을 보냅니다...")
        time.sleep(3)
        
        msg = "아메리카노"
        client_soc.sendall(msg.encode('utf-8'))
        print(f"oss [Server] Sent: {msg}")

        # 연결 유지
        time.sleep(10)
        client_soc.close()
        server_socket.close()
        print("--- 서버 테스트 종료 ---")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

# --- 여기가 핵심입니다. 이 부분이 없으면 아무 일도 안 일어납니다. ---
if __name__ == '__main__':
    print("0. 메인 함수 실행")
    run_server()