import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# Carregando o conjunto de dados Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris.target_names[iris.target]

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Criando e treinando um classificador k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


iris_df['predictions'] = knn.predict(iris.data)


accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy:.2f}')

conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=iris.target_names)

print('Matriz de Confusão:')
print(conf_matrix)
print('\nRelatório de Classificação:')
print(class_report)

pca = PCA(n_components=3)
iris_3d = pca.fit_transform(iris.data)


iris_df['PCA1'] = iris_3d[:, 0]
iris_df['PCA2'] = iris_3d[:, 1]
iris_df['PCA3'] = iris_3d[:, 2]


fig = px.scatter_3d(iris_df, x='PCA1', y='PCA2', z='PCA3', color='predictions', symbol='species',
                    size_max=10, opacity=0.7, title='Previsões do Modelo k-NN - Iris Dataset')

# Exibindo o gráfico
fig.show()
