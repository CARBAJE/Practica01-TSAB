import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog, Tk
from scipy.stats import entropy

# Función sigmoide para mejorar el contraste
def sigmoid_contrast(image, alpha, beta):
    image = image.astype(np.float32)
    return 255 / (1 + np.exp(-alpha * (image - beta)))

# Función para calcular la entropía de una imagen
def image_entropy(image):
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256), density=True)
    return entropy(hist + 1e-10)  # Pequeño valor para evitar log(0)

# Función de evaluación (fitness) basada en entropía
def fitness_function(params, image):
    alpha, beta = params
    enhanced_image = sigmoid_contrast(image, alpha, beta)
    return -image_entropy(enhanced_image)  # Minimizar el negativo de la entropía para maximizarla

# Algoritmo bioinspirado para optimizar el contraste
def optimize_contrast(image):
    Np = 20  # Tamaño de la población
    Nvar = 2  # Número de variables (alpha, beta)
    Generations = 50  # Número de generaciones
    Pc = 0.9  # Probabilidad de cruce
    Pm = 0.1  # Probabilidad de mutación
    Nc = 2  # Parámetro de cruce
    Nm = 2  # Parámetro de mutación
    lb = np.array([0.01, 0])  # Límites inferiores de alpha y beta
    ub = np.array([1, 255])  # Límites superiores de alpha y beta

    # Inicialización de la población
    population = np.random.uniform(lb, ub, (Np, Nvar))
    for gen in range(Generations):
        fitness = np.array([fitness_function(ind, image) for ind in population])
        sorted_indices = np.argsort(fitness)
        parents = population[sorted_indices[:Np]]
        offspring = np.zeros((Np, Nvar))
        
        for i in range(0, Np - 1, 2):
            if np.random.rand() <= Pc:
                U = np.random.rand()
                for j in range(Nvar):
                    P1, P2 = parents[i, j], parents[i + 1, j]
                    beta_c = (U ** (1 / (Nc + 1))) if U <= 0.5 else (1 / (2 - U * 2)) ** (1 / (Nc + 1))
                    offspring[i, j] = 0.5 * ((P1 + P2) - beta_c * abs(P2 - P1))
                    offspring[i + 1, j] = 0.5 * ((P1 + P2) + beta_c * abs(P2 - P1))
            else:
                offspring[i, :], offspring[i + 1, :] = parents[i, :], parents[i + 1, :]
        
        for i in range(Np):
            for j in range(Nvar):
                if np.random.rand() <= Pm:
                    delta_q = np.random.uniform(-0.1, 0.1)
                    offspring[i, j] += delta_q * (ub[j] - lb[j])
                    offspring[i, j] = np.clip(offspring[i, j], lb[j], ub[j])
        
        all_individuals = np.vstack((population, offspring))
        all_fitness = np.array([fitness_function(ind, image) for ind in all_individuals])
        best_indices = np.argsort(all_fitness)[:Np]
        population = all_individuals[best_indices]

    # Selección de los 5 mejores parámetros
    best_params = population[:5]
    best_fitness = np.array([fitness_function(ind, image) for ind in best_params])
    
    return best_params, best_fitness

# Simulación de 5 ejecuciones
Tk().withdraw()
file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp;*.png;*.jpg;*.jpeg")])
if not file_path:
    print("No se seleccionó ninguna imagen.")
    exit()

image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
all_results = []

for run in range(5):
    print(f"\nEjecución {run + 1}")
    best_params, best_fitness = optimize_contrast(image)
    
    for i, (params, fitness) in enumerate(zip(best_params, best_fitness)):
        all_results.append([run + 1, i + 1, params[0], params[1], -fitness])  # Guardar fitness positivo

# Crear DataFrame con resultados
df = pd.DataFrame(all_results, columns=["Ejecución", "Rank", "Alpha", "Beta", "Fitness"])

# Mostrar tabla de estadísticas
stats = df.groupby("Ejecución")["Fitness"].agg(["max", "min", "mean", "std"]).reset_index()
print("\nEstadísticas de Fitness por ejecución:")
print(stats)

# Visualización de las 5 mejores imágenes de la última ejecución
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, (alpha, beta) in enumerate(best_params):
    enhanced_image = sigmoid_contrast(image, alpha, beta)
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    axes[i].imshow(enhanced_image, cmap='gray')
    axes[i].set_title(f"α: {alpha:.2f}, β: {beta:.2f}, F: {-best_fitness[i]:.2f}")
    axes[i].axis('off')

plt.show()
