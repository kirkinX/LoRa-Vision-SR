# ==============================================================================
# LORA-VISION-SR: COMPRESIÓN ESPECTRAL Y SÚPER-RESOLUCIÓN POR RADIOFRECUENCIA
# Licencia: Código Abierto (Apache 2.0 / BSD-3 - Libre de Regalías)
# ==============================================================================

# ------------------------------------------------------------------------------
# PARTE 1: LIBRERÍAS Y PARÁMETROS DE ENLACE TÁCTICO
# ------------------------------------------------------------------------------
import os
import time
import math
import cv2
import numpy as np

ARCHIVO_ORIGEN = "Foto_prueba.jpg"
MODELO_IA_PATH = "EDSR_x4.pb"

RES_LUMINANCIA = (128, 128)      # Canal Y (Nitidez y bordes -> IA x4 = 512x512)
RES_CROMINANCIA = (32, 32)       # Canales Cb/Cr (Color real ultra-ligero)
ESCALA_IA = 4                    # Factor de ampliación neuronal
CALIDAD_WEBP_Y = 42              # Compresión sin bloques para nitidez
CALIDAD_WEBP_COLOR = 30          # Compresión extrema para mapas de color
PAYLOAD_MAX_LORA = 220           # Bytes útiles por paquete estándar LoRa

# ------------------------------------------------------------------------------
# PARTE 2: NODO EMISOR (DESCOMPOSICIÓN ESPECTRAL Y-Cb-Cr PARA RADIO)
# ------------------------------------------------------------------------------
def empaquetar_para_radio(imagen_bgr, modo_color=True):
    """
    Separa la luz (Y) del color real (Cb, Cr) para enviar nitidez en alta prioridad
    y color real en micro-paquetes adicionales sin saturar el enlace LoRa.
    """
    ycrcb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2YCrCb)
    canal_y, canal_cr, canal_cb = cv2.split(ycrcb)

    # 1. Procesar Canal Y (Luminancia - 128x128 con realce de bordes)
    y_mini = cv2.resize(canal_y, RES_LUMINANCIA, interpolation=cv2.INTER_AREA)
    desenfoque = cv2.GaussianBlur(y_mini, (0, 0), 1.5)
    y_nitido = cv2.addWeighted(y_mini, 1.4, desenfoque, -0.4, 0)

    _, buf_y = cv2.imencode(".webp", y_nitido, [int(cv2.IMWRITE_WEBP_QUALITY), CALIDAD_WEBP_Y])
    bytes_y = buf_y.tobytes()

    bytes_cr, bytes_cb = b"", b""
    if modo_color:
        # 2. Procesar Canales de Color Real (32x32 - apenas unos cientos de bytes)
        cr_mini = cv2.resize(canal_cr, RES_CROMINANCIA, interpolation=cv2.INTER_AREA)
        cb_mini = cv2.resize(canal_cb, RES_CROMINANCIA, interpolation=cv2.INTER_AREA)

        _, buf_cr = cv2.imencode(".webp", cr_mini, [int(cv2.IMWRITE_WEBP_QUALITY), CALIDAD_WEBP_COLOR])
        _, buf_cb = cv2.imencode(".webp", cb_mini, [int(cv2.IMWRITE_WEBP_QUALITY), CALIDAD_WEBP_COLOR])
        bytes_cr = buf_cr.tobytes()
        bytes_cb = buf_cb.tobytes()

    paquete_radio = {
        "y": bytes_y,
        "cr": bytes_cr,
        "cb": bytes_cb,
        "modo_color": modo_color
    }

    total_bytes = len(bytes_y) + len(bytes_cr) + len(bytes_cb)
    paquetes_lora = math.ceil(total_bytes / PAYLOAD_MAX_LORA)
    return paquete_radio, total_bytes, paquetes_lora

# ------------------------------------------------------------------------------
# PARTE 3: NODO RECEPTOR (SÚPER-RESOLUCIÓN EDSR x4 + FUSIÓN DE COLOR REAL)
# ------------------------------------------------------------------------------
def reconstruir_en_base(paquete_radio):
    """Reconstruye el canal Y con IA EDSR x4 y fusiona los canales reales Cr/Cb."""
    t_inicio = time.time()

    # Decodificar canal de luminancia recibido
    y_recibido = cv2.imdecode(np.frombuffer(paquete_radio["y"], np.uint8), cv2.IMREAD_GRAYSCALE)
    destino = (RES_LUMINANCIA[0] * ESCALA_IA, RES_LUMINANCIA[1] * ESCALA_IA)

    # Aplicar Súper-Resolución IA sobre la luminancia
    metodo_usado = "EDSR-x4 (Red Neuronal Determinista)"
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        if not os.path.exists(MODELO_IA_PATH):
            raise FileNotFoundError(f"Falta {MODELO_IA_PATH}")
        sr.readModel(MODELO_IA_PATH)
        sr.setModel("edsr", ESCALA_IA)
        y_hd = sr.upsample(y_recibido)

        suave = cv2.GaussianBlur(y_hd, (0, 0), 1.0)
        y_hd = cv2.addWeighted(y_hd, 1.25, suave, -0.25, 0)
    except Exception as e:
        metodo_usado = f"Lanczos4 ({e})"
        y_hd = cv2.resize(y_recibido, destino, interpolation=cv2.INTER_LANCZOS4)

    # Si llegaron canales de color real, fusionarlos sobre la imagen HD
    if paquete_radio["modo_color"] and paquete_radio["cr"] and paquete_radio["cb"]:
        cr_rec = cv2.imdecode(np.frombuffer(paquete_radio["cr"], np.uint8), cv2.IMREAD_GRAYSCALE)
        cb_rec = cv2.imdecode(np.frombuffer(paquete_radio["cb"], np.uint8), cv2.IMREAD_GRAYSCALE)

        cr_hd = cv2.resize(cr_rec, destino, interpolation=cv2.INTER_CUBIC)
        cb_hd = cv2.resize(cb_rec, destino, interpolation=cv2.INTER_CUBIC)

        ycrcb_reconstruido = cv2.merge([y_hd, cr_hd, cb_hd])
        img_final = cv2.cvtColor(ycrcb_reconstruido, cv2.COLOR_YCrCb2BGR)
    else:
        img_final = cv2.cvtColor(y_hd, cv2.COLOR_GRAY2BGR)

    tiempo_ms = (time.time() - t_inicio) * 1000
    return y_recibido, img_final, metodo_usado, tiempo_ms

# ------------------------------------------------------------------------------
# PARTE 4: GENERADOR DE EVIDENCIA COMPARATIVA (PARA GITHUB Y AUDITORÍA)
# ------------------------------------------------------------------------------
def guardar_comparativa(img_orig_bgr, y_mini, img_final_bgr):
    """Genera panel comparativo: Original Color vs LoRa Raw vs IA Color Real."""
    tam = (512, 512)

    p1 = cv2.resize(img_orig_bgr, tam)
    p2 = cv2.cvtColor(cv2.resize(y_mini, tam, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    p3 = cv2.resize(img_final_bgr, tam)

    cv2.putText(p1, "1. ORIGINAL (COLOR)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(p2, "2. RADIO LORA RAW", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(p3, "3. IA EDSR + COLOR REAL", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    panel = np.hstack((p1, p2, p3))
    cv2.imwrite("comparativa_github_sr.jpg", panel)

# ------------------------------------------------------------------------------
# PARTE 5: EJECUCIÓN Y TELEMETRÍA DE ENLACE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("============================================================")
    print(" LORA-VISION-SR | TRANSMISIÓN ESPECTRAL A COLOR POR RADIO")
    print("============================================================")

    img_original = cv2.imread(ARCHIVO_ORIGEN)
    if img_original is None:
        print(f"[ERROR] No se encontró '{ARCHIVO_ORIGEN}'.")
        exit(1)

    peso_orig_kb = os.path.getsize(ARCHIVO_ORIGEN) / 1024
    print(f"[*] Origen: {ARCHIVO_ORIGEN} ({peso_orig_kb:.1f} KB)")

    # Emisión en Modo Reconocimiento (Luminancia + Color Real)
    paquete, total_bytes, num_paquetes = empaquetar_para_radio(img_original, modo_color=True)
    print(f"[+] Canal Y (Nitidez 128x128): {len(paquete['y'])} bytes")
    print(f"[+] Canales Cr+Cb (Color 32x32): {len(paquete['cr']) + len(paquete['cb'])} bytes")
    print(f"[+] Peso Total Radio: {total_bytes} bytes ({total_bytes/1024:.2f} KB) -> {num_paquetes} paquetes LoRa")

    # Recepción + IA + Fusión de Color
    y_mini, img_final, metodo, ms = reconstruir_en_base(paquete)
    cv2.imwrite("mejorada_color_512x512.jpg", img_final)
    guardar_comparativa(img_original, y_mini, img_final)

    print(f"[+] Motor IA: {metodo} | Tiempo: {ms:.2f} ms")
    print("[+] Evidencia generada: 'mejorada_color_512x512.jpg' y 'comparativa_github_sr.jpg'")