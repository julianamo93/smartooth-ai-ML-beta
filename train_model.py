import joblib
from sklearn.ensemble import RandomForestClassifier  
import numpy as np

# dados fictícios para treinamento
X = np.array([[30, 1, 1], [40, 2, 3], [35, 1, 2]])  
y = np.array([1, 0, 1])  # Exemplo de rótulos (labels)

# Criar e treinar o modelo
model = RandomForestClassifier()
model.fit(X, y)

# Salvar o modelo treinado no arquivo
joblib.dump(model, 'models/model.pkl')

print("Modelo treinado e salvo com sucesso!")
