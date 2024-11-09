from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from app.ml_models import get_dental_tips, get_recommendations, predict, generate_recommendation


# Carregar o modelo treinado
model = joblib.load('models/model.pkl')  # Verifique o caminho correto para o arquivo do modelo

# Funções fictícias para representar o comportamento das funções de dicas e recomendações
# Estas devem ser substituídas pelas suas funções reais, como get_dental_tips, etc.

def get_dental_tips(patient_id):
    # Lógica fictícia para retornar dicas odontológicas
    return {"tips": ["Escove os dentes duas vezes ao dia", "Use fio dental diariamente"]}

def get_recommendations(patient_id):
    # Lógica fictícia para retornar recomendações
    return {"recommendations": ["Tratamento de canal", "Limpeza profissional"]}

# Criar a aplicação Flask
app = Flask(__name__)

# Rota principal para testar se o servidor está funcionando
@app.route('/')
def home():
    return "Welcome to Smartooth AI"

# Rota para filtrar os procedimentos odontológicos
@app.route('/filter_procedures', methods=['GET'])
def filter_procedures():
    try:
        procedures = pd.read_csv('data/procedures.csv')  # Certifique-se de que o arquivo está no local correto
        return jsonify(procedures.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Rota para retornar dicas odontológicas baseadas no patient_id
@app.route('/dental_tips', methods=['GET'])
def dental_tips():
    patient_id = request.args.get('patient_id')
    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400
    
    tips = get_dental_tips(patient_id)  # Substitua por sua função real
    return jsonify(tips)

# Rota para retornar recomendações baseadas no patient_id
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        # Obtém os dados JSON da requisição
        data = request.get_json()
        
        # Verifica se os dados essenciais estão presentes na requisição
        if not data or 'age' not in data or 'history' not in data or 'severity' not in data:
            return jsonify({"error": "Missing required data: 'age', 'history', and 'severity'"}), 400
        
        # Chama a função de previsão passando os dados
        prediction = predict(data)
        
        # Garante que o valor da previsão seja extraído corretamente
        recommendation = generate_recommendation(prediction)
        
        # Retorna a previsão e a recomendação gerada
        return jsonify({
            'prediction': prediction,  # Exibe a previsão (0 ou 1)
            'recommendation': recommendation  # A recomendação gerada
        })
    
    except Exception as e:
        # Se houver algum erro, retorna a mensagem de erro
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)  # Roda a aplicação Flask em modo de depuração
