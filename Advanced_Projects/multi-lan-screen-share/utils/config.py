# Configuration constants for the screen share project

# Network settings
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT_UDP = 5001
DEFAULT_PORT_TCP = 5002

# Buffer sizes
UDP_BUFFER_SIZE = 65507  # Max UDP packet size
TCP_BUFFER_SIZE = 1024 * 1024  # 1MB for TCP

# Compression settings
COMPRESSION = 'zlib'  # Options: 'zlib', 'lz4', 'none'
JPEG_QUALITY = 70  # JPEG quality for encoding

# Frame settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 20
