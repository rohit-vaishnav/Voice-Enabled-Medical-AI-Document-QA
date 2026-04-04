"""
qr_generator.py
Generate a QR code for the web app URL.
"""

import qrcode
import tempfile
import os


def generate_qr_code(url: str, save_path: str = None) -> str:
    """
    Generate a QR code image for the given URL.

    Args:
        url: The URL to encode.
        save_path: Optional file path to save. Uses temp file if not given.

    Returns:
        Path to the saved QR code PNG image.
    """
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=8,
        border=3,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="#1a3a5c", back_color="white")

    if save_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        save_path = tmp.name
        tmp.close()

    img.save(save_path)
    return save_path
