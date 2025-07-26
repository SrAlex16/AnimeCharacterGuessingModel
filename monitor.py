import tensorflow as tf
import psutil
import time
import matplotlib.pyplot as plt
import os
import numpy

# Intentar importar pynvml para monitoreo de GPU
try:
    from pynvml import *
    GPU_MONITORING = True
except ImportError:
    print("Advertencia: pynvml no encontrado. El monitoreo de GPU NVIDIA no estará disponible.")
    print("Instala con: pip install nvidia-ml-py")
    GPU_MONITORING_AVAILABLE = False
except NVMLError as error:
    print(f"Advertencia: Error al inicializar NVML para monitoreo de GPU: {error}")
    GPU_MONITORING_AVAILABLE = False

# --- Callback personalizado para Monitoreo de recursos ---
class ResourceMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_interval_batches=100):
        super().__init__()
        self.log_interval_batches = log_interval_batches
        self.batch_count = 0
        self.start_time = time.time()
        self.gpu_handle = None
        
        # Listas para almacenar las métricas de recursos
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = []
        self.gpu_memory_usage = []
        self.timestamps = []

        if GPU_MONITORING_AVAILABLE:
            try:
                nvmlInit()
                # Asume la primera GPU (índice 0)
                # Puedes iterar sobre nvmlDeviceGetCount() si tienes múltiples GPUs
                self.gpu_handle = nvmlDeviceGetHandleByIndex(0) 
                print("\nMonitoreo de GPU NVIDIA iniciado.")
            except NVMLError as error:
                print(f"Error al iniciar monitoreo de GPU: {error}")
                self.gpu_handle = None

    def on_train_begin(self, logs=None):
        print("\nIniciando monitoreo de recursos...")
        self.start_time = time.time()
        self._log_resources("Inicio de entrenamiento")

    def on_train_end(self, logs=None):
        self._log_resources("Fin de entrenamiento")
        if self.gpu_handle:
            nvmlShutdown()
            print("Monitoreo de GPU NVIDIA finalizado.")

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.log_interval_batches == 0:
            self._log_resources(f"Después del lote {self.batch_count}")

    def on_epoch_end(self, epoch, logs=None):
        self._log_resources(f"Fin de la época {epoch + 1}")

    def _log_resources(self, stage):
        current_time = time.time() - self.start_time # Tiempo transcurrido
        self.timestamps.append(current_time)

        cpu_percent = psutil.cpu_percent(interval=None) 
        self.cpu_usage.append(cpu_percent)

        ram_percent = psutil.virtual_memory().percent
        self.ram_usage.append(ram_percent)
        
        log_message = f"[{time.strftime('%H:%M:%S')}] {stage} - CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}%"
        
        if self.gpu_handle:
            try:
                utilization = nvmlDeviceGetUtilizationRates(self.gpu_handle)
                memory_info = nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util = utilization.gpu
                gpu_mem_used_mb = memory_info.used / (1024**2)
                gpu_mem_total_mb = memory_info.total / (1024**2)
                gpu_mem_percent = (gpu_mem_used_mb / gpu_mem_total_mb) * 100 if gpu_mem_total_mb > 0 else 0
                
                self.gpu_usage.append(gpu_util)
                self.gpu_memory_usage.append(gpu_mem_percent) # Guardamos el porcentaje de memoria
                
                log_message += f" | GPU: {gpu_util:.1f}% | GPU Mem: {gpu_mem_used_mb:.1f}MB ({gpu_mem_percent:.1f}%)"
            except NVMLError as error:
                log_message += f" | Error GPU: {error}"
                self.gpu_usage.append(np.nan) # Añadir NaN si hay error para mantener la longitud
                self.gpu_memory_usage.append(np.nan)
        else:
            self.gpu_usage.append(np.nan)
            self.gpu_memory_usage.append(np.nan)
        
        print(log_message)

# --- Funciones de Graficación ---

def plot_training_history(history, save_path=None):
    """
    Grafica la precisión y la pérdida del entrenamiento y la validación.
    """
    plt.figure(figsize=(12, 5))

    # Gráfico de Precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True)

    # Gráfico de Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico de historial de entrenamiento guardado en: {save_path}")
    plt.show()

def plot_resource_usage(monitor_callback: ResourceMonitorCallback, save_path=None):
    """
    Grafica el uso de CPU, RAM y GPU (si está disponible) durante el entrenamiento.
    """
    if not monitor_callback.timestamps:
        print("No hay datos de uso de recursos para graficar.")
        return

    plt.figure(figsize=(15, 7))

    # Gráfico de uso de CPU
    plt.subplot(1, 3, 1)
    plt.plot(monitor_callback.timestamps, monitor_callback.cpu_usage, label='Uso de CPU (%)', color='blue')
    plt.title('Uso de CPU durante el Entrenamiento')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Uso de CPU (%)')
    plt.grid(True)
    plt.legend()

    # Gráfico de uso de RAM
    plt.subplot(1, 3, 2)
    plt.plot(monitor_callback.timestamps, monitor_callback.ram_usage, label='Uso de RAM (%)', color='green')
    plt.title('Uso de RAM durante el Entrenamiento')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Uso de RAM (%)')
    plt.grid(True)
    plt.legend()

    # Gráfico de uso de GPU (si disponible)
    if GPU_MONITORING_AVAILABLE and monitor_callback.gpu_handle:
        plt.subplot(1, 3, 3)
        plt.plot(monitor_callback.timestamps, monitor_callback.gpu_usage, label='Uso de GPU (%)', color='red')
        plt.plot(monitor_callback.timestamps, monitor_callback.gpu_memory_usage, label='Uso de Memoria GPU (%)', color='purple', linestyle='--')
        plt.title('Uso de GPU durante el Entrenamiento')
        plt.xlabel('Tiempo (segundos)')
        plt.ylabel('Uso (%)')
        plt.grid(True)
        plt.legend()
    else:
        # Si no hay GPU o monitoreo, dejamos este subplot vacío o con un mensaje
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, "Monitoreo de GPU no disponible", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title('Uso de GPU')
        plt.axis('off') # Ocultar ejes

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico de uso de recursos guardado en: {save_path}")
    plt.show()

# Bloque de ejecución principal para probar el módulo directamente
if __name__ == "__main__":
    print("Este script está diseñado para ser importado. Ejecutando una prueba simple de graficación.")
    
    # Datos de prueba para el historial de entrenamiento
    class MockHistory:
        def __init__(self):
            self.history = {
                'accuracy': [0.7, 0.75, 0.8, 0.82, 0.85],
                'val_accuracy': [0.68, 0.72, 0.77, 0.79, 0.81],
                'loss': [0.5, 0.45, 0.4, 0.38, 0.35],
                'val_loss': [0.52, 0.48, 0.43, 0.41, 0.39]
            }
    mock_history = MockHistory()
    plot_training_history(mock_history, save_path="test_history_plot.png")

    # Datos de prueba para el monitoreo de recursos
    mock_monitor = ResourceMonitorCallback(log_interval_batches=1)
    for i in range(5):
        mock_monitor.timestamps.append(i * 10) # cada 10 segundos
        mock_monitor.cpu_usage.append(np.random.uniform(30, 70))
        mock_monitor.ram_usage.append(np.random.uniform(40, 60))
        if GPU_MONITORING_AVAILABLE:
            mock_monitor.gpu_usage.append(np.random.uniform(10, 90))
            mock_monitor.gpu_memory_usage.append(np.random.uniform(20, 80))
    plot_resource_usage(mock_monitor, save_path="test_resources_plot.png")