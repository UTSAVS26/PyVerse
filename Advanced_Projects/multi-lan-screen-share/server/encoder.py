
from io import BytesIO
import zlib
try:
    import lz4.frame
    _has_lz4 = True
except ImportError:
    _has_lz4 = False
from utils import config

def encode_jpeg(image, quality=None):
    buf = BytesIO()
    image.save(buf, format='JPEG', quality=quality or config.JPEG_QUALITY)
    return buf.getvalue()

def compress_data(data, method=None):
    method = method or config.COMPRESSION
    if method == 'zlib':
        return zlib.compress(data)
    elif method == 'lz4' and _has_lz4:
        return lz4.frame.compress(data)
    else:
        return data

def decompress_data(data, method=None):
    method = method or config.COMPRESSION
    try:
        if method == 'zlib':
            return zlib.decompress(data)
        elif method == 'lz4' and _has_lz4:
            return lz4.frame.decompress(data)
        else:
            return data
    except Exception as e:
        raise RuntimeError(f"Decompression failed with {method}: {e}") from e
