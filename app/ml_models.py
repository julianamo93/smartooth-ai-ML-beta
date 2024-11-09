import random
import joblib
import numpy as np

# Carregar o modelo treinado
model = joblib.load('models/model.pkl')

def get_dental_tips(patient_id):
    # Dicas de cuidados dentários simples, baseadas no histórico
    tips = [
        "Escove seus dentes duas vezes ao dia.",
        "Use fio dental regularmente.",
        "Consuma menos açúcar para evitar cáries."
    ]
    return random.sample(tips, 2)  # Retorna 2 dicas aleatórias para simplicidade

def get_recommendations(patient_id):
    # Recomendações simples para o paciente
    recommendations = [
        "Agende uma consulta de limpeza.",
        "Faça um check-up para verificar o estado das suas gengivas.",
        "Tente usar um creme dental com flúor."
    ]
    return random.sample(recommendations, 2)  # Retorna 2 recomendações aleatórias para simplicidade

def predict(input_data):
    """
    Função para realizar a previsão com o modelo treinado.
    O modelo espera um dicionário com 'age', 'history' e 'severity' como entradas.
    """
    # Adaptando os dados de entrada para o formato esperado pelo modelo
    features = np.array([input_data['age'], input_data['history'], input_data['severity']]).reshape(1, -1)

    # Realizando a previsão
    prediction = model.predict(features)
    return prediction

def generate_recommendation(prediction):
    """
    Gera uma recomendação baseada na previsão do modelo.
    Se a previsão for 1, recomenda tratamento preventivo.
    Caso contrário, sugere acompanhamento regular com o dentista.
    """
    if prediction[0] == 1:
        return "Recomendamos um tratamento preventivo."
    else:
        return "Mantenha acompanhamento regular com o dentista."
