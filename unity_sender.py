import socket

# UDP configuration
UDP_IP = "127.0.0.1"  # Unity IP (localhost for testing)
UDP_PORT = 12345
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_va_to_unity(v_raw, a_raw):
    """
    Send valence and arousal to Unity via UDP.
    """
    try:
        message = f"{v_raw},{a_raw}".encode('utf-8')
        sock.sendto(message, (UDP_IP, UDP_PORT))
        print(f"Sent VA to Unity: {v_raw}, {a_raw}")
    except Exception as e:
        print(f"UDP send error: {e}")