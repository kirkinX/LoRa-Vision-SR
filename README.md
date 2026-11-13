# LoRa-Vision-SR: Transmisión Espectral de Imágenes y Súper-Resolución por Radiofrecuencia

**LoRa-Vision-SR** es un motor de compresión extrema y reconstrucción visual determinista diseñado para enlaces de comunicación críticos de ultra-bajo ancho de banda (radiofrecuencia **LoRa 433/915 MHz**, telemetría satelital y sistemas autónomos sin cobertura celular/Wi-Fi).

![Comparativa Visual](comparativa_github_sr.jpg)

## Arquitectura Técnica

El sistema resuelve el cuello de botella físico de las redes LPWAN (donde cada paquete útil transporta ~220 bytes) dividiendo la imagen en el espacio de color **YCbCr** antes de transmitir:

1. **Modo Patrulla Rápida (Solo Luminancia `Y` - B/N):**
   * Extrae únicamente el canal de intensidad lumínica (`128x128 px`), aplica realce de micro-contrastes (*Unsharp Mask*) y codifica en **WebP** (~1.8 - 2.2 KB, equivalente a **9–10 paquetes LoRa**).
2. **Modo Reconocimiento Espectral (`Y + Cr + Cb` - Color Real):**
   * Envía micro-paquetes adicionales de crominancia (`32x32 px`, ~400 bytes extra) que preservan los colores reales de la escena sin depender de colorización generativa.
3. **Reconstrucción Determinista en Estación Base (`EDSR x4`):**
   * La estación receptora decodifica los paquetes y escala la imagen a **`512x512 px`** mediante la red neuronal profunda **EDSR** (*Enhanced Deep Residual Networks*), recuperando bordes nítidos sin introducir alucinaciones visuales.

## Características Principales
* **Cero Alucinaciones:** Emplea redes deterministas de súper-resolución (EDSR/FSRCNN) aptas para auditoría, seguridad perimetral y robótica.
* **Transmisión Progresiva:** Permite visualizar primero la estructura en escala de grises y fusionar el color real bajo demanda.
* **100% Libre de Regalías:** Construido exclusivamente sobre librerías y pesos bajo licencias permisivas (**Apache 2.0 / BSD-3-Clause**).

## Instalación y Uso Rápido

```bash
# 1. Instalar dependencias
pip install opencv-contrib-python numpy

# 2. Descargar pesos del modelo determinista EDSR x4
wget -O EDSR_x4.pb https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb

# 3. Ejecutar prueba de compresión y reconstrucción
python3 lora_vision_sr.py
```
