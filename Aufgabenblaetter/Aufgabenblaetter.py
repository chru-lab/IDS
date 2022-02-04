#!/usr/bin/env python
# coding: utf-8

# # Aufgabenblatt 01

# ## Aufgabe 1
# a) Gegeben sind drei verschiedene Regressionsgeraden (rot-gestrichelte Linie). Bestimmen Sie für jede der Geraden A, B und C die beiden Parameter $\beta_0$ und $\beta_1$ durch Ablesen aus der Grafik.
# 
# ![image.png](attachment:image.png)

# In[1]:


get_ipython().run_line_magic('env', 'OMP_NUM_THREADS=4')


# ### Lösung
# - A: $\beta_0$= 1 , $\beta_1$= 1
# - B: $\beta_0$= 1, $\beta_1$= -1
# - C: $\beta_0$= -1, $\beta_1 = 0$ 
# 

# b) Gegeben sind folgende Datenerhebungen A, B und C mit den zugehörigen Datenpunkten. Entscheiden Sie für jede Erhebung, ob eine Analyse mittels linearer Regression möglich ist und begründen Sie kurz Ihre Antwort, falls dies nicht geht.
# 
# ![image.png](attachment:image.png)

# ### Lösung
# 
# - A: Keine Regression möglich, da kein zusammenhang der Variablen **(Musterlösung, steigung = Unendlich)**
# - B: Lineare Regression möglich 
# - C: Keine Regression möglich, da kein Linearer zusammenhang **(Musterlösung $R^2$ ist tief)**

# ## Aufgabe 2
# 
# Der „California Housing“-Datensatz aus dem Paket sklearn.datasets beinhaltet
# verschiedene Informationen aus 20.640 Haushalten in Kalifornien. Laden Sie diesen Datensatz
# und nutzen Sie daraus die Spalten 2 (durchschnittliche Anzahl an Räumen) und 3
# (durchschnittliche Anzahl an Schlafzimmern) für eine Regressionsanalyse. Gehen Sie dazu wie
# folgt vor: 

# In[2]:


# Vorbereitung Aufgabe 
from sklearn import datasets 
data = datasets.fetch_california_housing() 
data_x = data.data[:,2]
data_y = data.data[:,3] 
data_x = data_x.reshape((-1,1)) ## hinzugefügt
x_data = data_x
y_data = data_y


# In[3]:


data_x


# Teilen Sie die Daten im Verhältnis 80:20 in Trainings- und Testdaten auf und führen Sie damit
# eine Regressionsanalyse durch, um festzustellen, ob es einen linearen Zusammenhang
# zwischen Anzahl an Räumen und Anzahl an Schlafzimmern gibt. Bestimmen Sie für den
# Testdatensatz zudem den MSE sowie R2
# -Wert und plotten Sie Ihre Ergebnisse in einem
# Diagramm. 

# ### Lösung

# In[4]:


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
import seaborn as sns  # Schönere Grafiken
import numpy as np
from sklearn.linear_model import LinearRegression as lr


# In[5]:


split=int(data_x.size/100*20)
print(split)


# In[6]:


#split in train und test 
data_x.size
x_train = x_data[:-split]
x_test = x_data[-split:]
y_train = y_data[:-split]
y_test = y_data[-split:]


# In[7]:


model = lr()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
plt.plot(x_test,y_test,'ro')
plt.plot(x_test,y_pred,'b-')
plt.show()


# In[8]:


r2_score(y_test, y_pred)


# In[9]:


mse(y_test, y_pred, squared=True) 


# # Aufgabenblatt 2

# ## Aufgabe 1
# a) Gegeben ist folgende Tabelle mit n = 12 Werten der tatsächlichen Klassifikation y und der
# durch k-Nächste-Nachbarn vorhergesagten Klassifikation ŷ eines Datensatzes (Klassen 0
# und 1). Bestimmen Sie die Treffsicherheit des Algorithmus. 
# 
# b) Wie gross ist demnach die Fehlinterpretationsrate (Hamming-Verlust)?
# ![image.png](attachment:image.png)

# In[10]:


# a) Treffsicherheit berechnen
treffsicherheit = 1/12*(1 + 1 + 1 + 1 + 0 + 1 + 0 + 0 + 1 + 1 + 0 + 1)
print(treffsicherheit,"oder", str(treffsicherheit*100)+"%")
# b Hamming Verlust
print(1-treffsicherheit,"oder", str(((1-treffsicherheit)*100))+"%")


# ## Aufgabe 2 
# Die Datei „wine.txt“ in Moodle enthält drei Spalten des zugehörigen Datensatzes aus dem
# Paket sklearn.datasets. Die ersten beiden Spalten entsprechen 2D-Koordinaten (x und y),
# die dritte Spalte entspricht der tatsächlichen Klassifikation der Punkte.
# 
# a) Visualisieren Sie die Datei als Streudiagramm (Scatter Plot). 

# In[11]:


data = np.loadtxt("wine.txt", delimiter=",")


# In[12]:


# Daten Plotten 
#plt.axis([10,80,0,120])
colors = ['red', 'blue', 'green'] # Array mit den beiden farben 
# Farbcodierungen in Colormap speichern 
import matplotlib.colors 
cmap = matplotlib.colors.ListedColormap(colors)
plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap=cmap)
plt.show()
cmap


# - b) Ermitteln Sie anhand der Ellenbogen-Methode den optimalen k-Wert für eine Clusterbildung gemäss k-Mitten-Algorithmus und zeichnen Sie die zugehörige(Trägheits)Kurve. 

# In[13]:


from sklearn.cluster import KMeans
x_data = data[:,:2] # Spalten 0 und 1 -> Koordinaten
y_data = data[:,2] # Spalte 2 -< korrekte Klassifikation
inert = []
for k in range(1,11):
    model = KMeans(k)
    model.fit(x_data)
    inert.append(model.inertia_)
x = np.linspace(1,10,10)
plt.plot(x, inert, 'b-')
plt.plot(x, inert, 'bo')
plt.show() 


# Bester wert ist k = 3

# - c) Bestimmen Sie für den aus b) ermittelten optimalen k-Wert eine Vorhersage (fit_predict) der Klassenzugehörigkeit aller Punkte der Datei „wine.txt“ und visualisieren Sie das Ergebnis als Streudiagramm. Wie gut stimmt die vorhergesagte Klassifikation – rein optisch betrachtet – mit dem tatsächlichen Ergebnis aus a) überein? 

# In[14]:


model = KMeans(n_clusters=3, max_iter=300)
# klassifiziere mir meine Daten 
y_pred = model.fit_predict(data)
colors = ['green', 'red', 'blue']
cmap = matplotlib.colors.ListedColormap(colors)
plt.scatter(data[:,0], data[:,1], c=y_pred, cmap=cmap)
plt.show()


# # Aufgabenblatt 3
# 
# ## Aufgabe 1
# Gegeben sind folgende Datensätze A, B und C sowie die ersten beiden (nicht
# massstabsgetreuen) Komponenten PC 1 und PC 2 einer Hauptkomponentenzerlegung.
# Welche der drei Datensätze liefern mittels Principal Component Analysis beide
# Hauptkomponenten? 
# 
# ![image.png](attachment:image.png)

# Datensatz B liefert beide Hauptkomponenten, da sich diese auf beide Achsen projezieren lässt

# ## Aufgabe 2
# Gegeben ist folgende prozentuale Verteilung der Varianz auf die Hauptkomponenten (PC)
# eines 8-dimensionalen Datensatzes gemäss Principal Component Analysis. Zeichnen Sie die 
# Verteilung der Varianz in Python als kumulative Summe (d.h. aufsummiert) in ein Diagramm
# (x-Achse: Hauptkomponenten; y-Achse: Varianz [%]). Überlegen Sie ausserdem, auf wie viele
# Dimensionen sich der Datensatz praktisch ohne Verlust reduzieren lässt. 
# 
# ![image.png](attachment:image.png)

# In[15]:


varianz = [0., 80., 18., 1., 0.25, 0.25, 0.25, 0.25, 0.]
#x = np.linspace(0, 8, 9) oder einfach
x = [0., 1., 2., 3., 4., 5., 6., 7., 8.]
plt.xlabel('Hauptkomponenten')
plt.ylabel('Varianz [%]') 
plt.axis([0,8,0,110]) 
plt.plot(x, np.cumsum(varianz), 'r-') 
print(x)
plt.show()


# Der Datensatz lässt sich praktisch ohne Verluste auf zwei Hauptkomponenten reduzieren, da dies 98% der Varianz abdeckt (80+18)

# ## Aufgabe 3
# Der Iris-Datensatz aus dem Paket sklearn.datasets beinhaltet 150 Schwertlilien, die nach
# vier unterschiedlichen Merkmalen klassifiziert wurden. Laden Sie zunächst den Datensatz
# iris, der u.a. die Teile iris.data (Grösse 150x4) mit den Merkmalswerten sowie
# iris.target (Grösse 150x1) mit den korrekten Klassifikationen enthält. 

# In[16]:


from sklearn import datasets
iris = datasets.load_iris() 


# In[17]:


iris


# a) Führen Sie mittels PCA eine Reduktion des vierdimensionalen Merkmalraumes
# (iris.data) auf zwei Dimensionen durch und visualisieren Sie das Ergebnis als
# Streudiagramm (Scatter Plot). 

# In[18]:


from sklearn.decomposition import PCA
model = PCA(n_components=2) # reduktion auf zwei komponenten
# Projiziere die daten gleich anhand der Projektion
data_proj = model.fit_transform(iris.data)
# data_proj hat keine x,y sondern die Projizierten koordinaten aller punkte auf der x-achse und alternativ aller punkte auf der X-achse
model.components_ # liefert die Vektoren, aber nicht transponiert 


# In[19]:


iris.target.astype(float)


# In[20]:


np.transpose(model.components_)
np.dot(np.transpose(model.components_)[:,0],np.transpose(model.components_)[:,1]) # sollte 0 sein
model.explained_variance_ratio_ 


# In[21]:


data_proj[:,0]


# In[22]:


# Vieviel % der Information steckt entlang der jeweiligen Komponenten
#colors = ['green', 'red', 'blue']
#cmap = matplotlib.colors.ListedColormap(colors)
plt.scatter(data_proj[:,0], data_proj[:,1], c=iris.target) # plotten
plt.show()


# b) Wie verteilt sich die Varianz des dimensionsreduzierten Datensatzes auf die beiden
# Hauptkomponenten? Ist das ein brauchbares Ergebnis? 

# In[23]:


np.sum(model.explained_variance_ratio_)


# 98% der Varianz fällt auf die ersten beiden Hauptkomponenten, der Verlust ist minimal

# c) Bestimmen Sie mittels k-Mitten-Algorithmus die zugehörigen Cluster der
# dimensionsreduzierten Daten und visualisieren Sie das Ergebnis als Streudiagramm. Wie
# gut stimmt die vorhergesagte Klassifikation – rein optisch betrachtet – mit dem
# tatsächlichen Ergebnis aus a) überein? 

# In[24]:


# Medell erzeugen (default=8, iterationsgrenze 300 )
model = KMeans(n_clusters=3, max_iter=300)
# klassifiziere mir meine Daten 
y_pred = model.fit_predict(data_proj)
colors = ['red', 'blue', 'green']
cmap = matplotlib.colors.ListedColormap(colors)
plt.scatter(data_proj[:,0], data_proj[:,1], c=y_pred, cmap=cmap)
#plt.scatter(data[:,0], data[:,1], c=y_pred)
plt.show()


# Klassifikation ist nicht schlecht. Rechts wurden einige Punkte falsch zugeordnet

# # Aufgabenblatt 4
# 
# ## Aufgabe 1 
# Gegeben sind folgende Vektoren 
# ![image.png](attachment:image.png)
# Zeigen Sie rechnerisch und mittels Python, welche Paarungen obiger Vektoren orthogonal
# sind, d.h. senkrecht aufeinander stehen. 

# In[25]:


u = [3,4,0,1]
v = [1,0,-1,2]
w = [4,-3,4,0]


# In[26]:


print(np.dot(u, v))
print(np.dot(v, w))
print(np.dot(u, w))


# $\vec{u}$ und $\vec{w}$, sowie $\vec{v}$ und $\vec{w}$ stehen senkrecht aufeinander, da ihre skalarprodukte null sind 

# Rechnerische Lösung aus dem Aufgabenblatt (mir wahr nicht klar, was er da wollte:
# ![image-2.png](attachment:image-2.png)

# ## Aufgabe 2
# Gegeben ist folgende Matrix 
# $$A = \left(\begin{array}{rrrr} 
# 3 & 0 & 4 & 1 \\ 
# 2 & 1 & 3 & 2 \\ 
# \end{array}\right)$$
# 
# Bestimmen Sie die Transponierte $A^T$ und prüfen Sie Ihr Ergebnis mittels Python. 

# In[27]:


a = [3,0,4,1,2,1,3,2]
a = np.reshape(a,(2,4))
a


# Transponierte $A^T$: 
# $$A^T = \left(\begin{array}{rr} 
# 3 & 2 \\
# 0 & 1 \\ 
# 4 & 3 \\
# 1 & 2 \\ 
# \end{array}\right)$$

# In[28]:


a_t = np.transpose(a)
a_t


# b) Welche Dimension benötigt ein Vektor $\vec{v}$ für die Matrix-Vektor-Multiplikation $A\cdot \vec{v}$ und $A^T\cdot \vec{v}$ für vorbezeichnete Matrix? In welchem der beiden Fälle erfolgt eine
# Dimensionsreduktion bzw. eine Dimensionserhöhung? 

# - Für $A\cdot \vec{v}$ benötigt der Vektor eine Dimension von 4, es erfolgt eine Dimensionsreduktion
# - Für $A^T\cdot \vec{v}$ benötigt der Vektor eine Dimension von 2, es erfolgt eine Dimensionserhöhung

# c) Gegeben sind folgende Vektoren:
# ![image.png](attachment:image.png)
# 
# Berechnen Sie mittels Python die beiden Matrix-Vektor-Multiplikationen $A\cdot \vec{u}$ und $A^T\cdot \vec{v}$
# [Zur Übung können Sie die beiden Matrix-Vektor-Multiplikationen auch selbst berechnen.] 

# In[29]:


u = [2,0,4,1]
v = [1,2]


# In[30]:


np.dot(a, u)


# In[31]:


np.dot(a_t, v)


# ## Aufgabe 3
# Gegeben sind folgende Vektoren 
# ![image.png](attachment:image.png)
# a) Berechnen Sie das dyadische Produkt und kontrollieren Sie Ihr Ergebnis in Python
# auf *zwei unterschiedliche* Arten. 

# In[32]:


u = np.transpose([2,4,1])
v = np.transpose([3,1,3,2])
w = np.transpose([5,2])
u


# In[33]:


np.outer(u,v)


# In[34]:


np.einsum('i,j->ij',u,v) 


# b) Berechnen Sie den Tensor X in Python als äusseres Produkt der drei Vektoren $\vec{w} \circ \vec{u} \circ \vec{v}$.

# In[35]:


X = np.einsum('i,j,k -> ijk', w,u,v)
X


# ## Aufgabe 4
# Gegeben sind folgende Matrizen
# $$A = \left(\begin{array}{rrr} 
# 1 & 2 & 3  \\ 
# 4 & 5 & 6  \\ 
# \end{array}\right),
# B = \left(\begin{array}{rrr} 
# 7 & 8 & 8  \\ 
# 10 & 11 & 12  \\ 
# \end{array}\right)$$.
# 
# Berechnen Sie in Python das Kronecker-Produkt $A\otimes B$, das Khatri-Rao-Produkt $A\odot B$ und
# das Hadamard-Produkt $A\ast B$. 
# - Nutzen Sie das Modul tensorly für Ihre Berechnung.
# - Tipp: Erzeugen Sie sich die Matrizen als Reihung mittels np.arange( ) und ändern Sie deren Gestalt mittels reshape((#zeilen,#spalten)). 

# In[36]:


A = np.reshape([1,2,3,4,5,6],(2,3))
A


# In[37]:


B = np.reshape([7,8,9,10,11,12],(2,3))
B


# In[38]:


import tensorly as tl
tl.tenalg.kronecker((A, B)) # Kronecker-Produkt


# In[39]:


tl.tenalg.khatri_rao((A, B)) # Khatri-rao-Produkt


# In[40]:


A*B # Hadamard-Produkt


# # Aufgabenblatt 5: Tensorfakturierung
# ## Aufgabe 1
# Erzeugen Sie sich einen 3-Mode-Tensor $\chi$ der Grösse 5 x 2 x 5 mit den Zahlen von 1 bis 50.
# ```python
# X = np.arange(50).reshape((5,2,5))
# X = X + 1
# ```
# Entfalten Sie den Tensor mithilfe des Pakets tensorly nach allen drei Moden. 

# In[41]:


X = np.arange(50).reshape((5,2,5))
X = X + 1
X


# In[42]:


tl.unfold(X, 0)


# In[43]:


tl.unfold(X, 1)


# In[44]:


tl.unfold(X, 2)


# # Aufgabe 2
# 
# Erzeugen Sie sich einen 3-Mode-Tensor $\chi$ der Grösse 30x30x30 bestehend aus
# Zufallszahlen. 
# ```X = np.random.rand(30,30,30) ```
# Führen Sie sukzessive eine Tucker-Zerlegung von mit Rang $r \in [1, 50]$ gemäss

# In[45]:


X = np.random.rand(30,30,30)
X


# In[46]:


from tensorly.decomposition import tucker ## Tucker faktorisierung
err = [] 
for r in range(1,51):
    G, fac = tucker(X, (r, r, r))
    X_rec = tl.tucker_to_tensor((G, fac))
    err.append(tl.norm(X - X_rec)) 
plt.plot(err, 'r-')
plt.show() 
tl.norm(X - X_rec)


# **Aus Musterlösung**: *Es fällt auf, dass der Fehler stetig abnimmt und etwa ab Rang 28 [Anmerkung: Da es sich
# um einen Zufallstensor handelt, kann Ihre Kurve leicht anders aussehen] bei Null liegt. Das
# heisst, etwa ab Rang 28 lässt sich keine sichtbare Genauigkeitssteigerung mehr erzielen,
# die Faktorisierung ist damit nahezu perfekt.*

# # Aufgabe 5
# - Nicht gemacht -> zu aufwändig 
