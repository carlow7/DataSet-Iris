import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import plotly.express as px
from sklearn.decomposition import PCA

# Carregando o conjunto de dados Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris.target_names[iris.target]

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

# Avaliando a precisão do modelo k-NN
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Acurácia do modelo k-NN: {accuracy_knn:.2f}')

# Usando k-Means para clustering
kmeans = KMeans(n_clusters=3, random_state=42)
iris_df['cluster'] = kmeans.fit_predict(iris.data)

# Reduzindo a dimensionalidade para visualização
pca = PCA(n_components=2)
iris_2d = pca.fit_transform(iris.data)

# Adicionando os resultados do clustering ao DataFrame
iris_df['PCA1'] = iris_2d[:, 0]
iris_df['PCA2'] = iris_2d[:, 1]

# Criando um gráfico interativo com Plotly para visualizar os clusters
fig = px.scatter(iris_df, x='PCA1', y='PCA2', color='cluster', symbol='species',
                 title='k-Means Clustering (2D Projection with PCA)',
                 labels={'cluster': 'Cluster', 'species': 'Species'})

# Exibindo o gráfico
fig.show()
