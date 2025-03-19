import numpy as np
import random
import csv
import cv2
linf = [0,0]
lsup = [10,1]
pc = .90    #Probabilidad de cruza
nc = 2  #amplitud cruza 
nm = 20 #amplitud mutacion
pm = 0.03   # Probabilidad de mutar
tam_pob = 50  # Tamaño de la población
generations = 20  # Número de generaciones

img = cv2.imread('bioinspirados\Original_Medica6R.png', 0)
img_normalized = img / 255



def generar_poblacion():
    return [[random.uniform(linf[i], lsup[i]) for i in range(len(linf))] for _ in range(tam_pob)]

def seleccion_padres(poblacion):
    padres = []
    for _ in range(2):
        torneo = random.sample(poblacion, 3)
        ganador = min(torneo, key=aptitud)
        padres.append(ganador)
    return padres


def aptitud(individuo):
    img_sigmoid = 1/(1 + np.exp(individuo[0]*(img_normalized-individuo[1])))
    # Función de entropia    
    # hist = cv2.calcHist([img_sigmoid.astype(np.float32)], [0], None, [256], [0, 1])
    # hist /= hist.sum()
    # Evitar log(0) eliminando ceros
    # hist = hist[hist > 0]
    # entropia = -np.sum(hist * np.log2(hist))
    # return -entropia
    # Función de desviacion estandart
    std_dev = np.std(img_sigmoid)  # Calculamos la desviación estándar
    return -std_dev  # Negativo porque el algoritmo minimiza



def cruza(padre_1, padre_2):

    if random.random() <= pc:
        u = random.random()
        h1,h2 = [], []
        for i in range(len(padre_1)):
            b = 1 + 2 * min(padre_1[i] - linf[i], lsup[i] - padre_2[i]) / max(abs(padre_2[i] - padre_1[i]), 1e-9)
            a = 2 - pow(abs(b),-(nc+1))
            if u <= 1/a:
                b = pow(u*a, 1/(nc+1))
            else:
                b = pow(1/(2-u*a),1/(nc+1))
            h1.append(.5*(padre_1[i]+padre_2[i]-b*abs(padre_2[i]-padre_1[i])))
            h2.append(.5*(padre_1[i]+padre_2[i]+b*abs(padre_2[i]-padre_1[i])))
    else:
        h1,h2 = padre_1,padre_2
        
    return h1,h2

def mutation(individuo):
    for i in range(len(individuo)):
        if random.random() <= pm:
            r = 0.4
            d = min(individuo[i]-linf[i],lsup[i]-individuo[i])/ (lsup[i]-linf[i])
            if r <= 0.5:
                d = pow(2*r+(1-2*r)*pow(1-d,(nm+1)), 1/(nm+1)) - 1
            else:
                d = 1 - pow(2*(1-r)+2*(r-0.5)*pow(1-d,(nm+1)), 1/(nm+1))
            individuo[i]=individuo[i]+d*(lsup[i]-linf[i])
        
    return individuo

best = []
for i in range(10):
    poblacion = generar_poblacion()
    mejores = []

    for gen in range(generations):
        hijos = []
        
        for _ in range(tam_pob // 2):
            padres = seleccion_padres(poblacion)
            h1, h2 = cruza(padres[0], padres[1])
            hijos.append(mutation(h1))
            hijos.append(mutation(h2))
        poblacion = hijos  # Reemplazo generacional
        
        mejor = min(poblacion, key=aptitud)
        mejores.append(mejor)
        #print(f"Generación {gen + 1}: Mejor solución: {mejor} con aptitud {aptitud(mejor)}")
    mejor = min(mejores, key=aptitud)
    print(f"ejecucion {i+1}Mejor solución: {mejor} con aptitud {aptitud(mejor)}")
    best.append(mejor)
mejor = min(best, key=aptitud)
peor = max(best, key=aptitud)
desviacion_estandar = np.std([aptitud(sol) for sol in best])

# Guardar en CSV
nombre_archivo = "resultados.csv"
with open(nombre_archivo, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Mejor Solución", "Aptitud Mejor", "Peor Solución", "Aptitud Peor", "Desviación Estándar"])
    writer.writerow([mejor, aptitud(mejor), peor, aptitud(peor), desviacion_estandar])


img_sigmoid = 1/(1 + np.exp(mejor[0]*(img_normalized-mejor[1])))
# transformar la imagen a color
# img_sigmoid_8bit = (img_sigmoid * 255).astype(np.uint8)
# img_colormap = cv2.applyColorMap(img_sigmoid_8bit, cv2.COLORMAP_JET)
img_resized = cv2.resize(img_sigmoid, (300, 300))  
cv2.imshow('imagen', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()