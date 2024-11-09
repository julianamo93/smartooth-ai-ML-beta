from flask import jsonify, request
import pandas as pd
from app.ml_models import get_dental_tips, get_recommendations, predict, generate_recommendation

def init_routes(app):
    @app.route('/')
    def home():
        return jsonify(message="Welcome to Smartooth AI")

    @app.route('/filter_procedures', methods=['GET'])
    def filter_procedures():
        try:
            procedures = pd.read_csv('data/procedures.csv')  # Verifique o caminho correto do arquivo CSV
            if procedures.empty:
                return jsonify({"error": "No procedures data available"}), 404
            return jsonify(procedures.to_dict(orient='records'))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/dental_tips', methods=['GET'])
    def dental_tips():
        patient_id = request.args.get('patient_id')
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        try:
            tips = get_dental_tips(patient_id)
            if not tips:
                return jsonify({"error": "No dental tips found for this patient."}), 404
            return jsonify(tips=tips)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/recommendations', methods=['GET'])
    def recommendations():
        patient_id = request.args.get('patient_id')
        if not patient_id:
            return jsonify({"error": "Missing patient_id"}), 400
        
        try:
            recommendations = get_recommendations(patient_id)
            if not recommendations:
                return jsonify({"error": "No recommendations found for this patient."}), 404
            return jsonify(recommendations=recommendations)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/predict', methods=['POST'])
    def predict_route():
        try:
            data = request.get_json()
            print("Received data:", data)  # Log dos dados recebidos

            if not data or 'age' not in data or 'history' not in data or 'severity' not in data:
                return jsonify({"error": "Missing required data: 'age', 'history', and 'severity'"}), 400

            # Chama a função de previsão passando os dados
            prediction = predict(data)
            
            # resultado da previsão seja um número simples
            if isinstance(prediction, (list, np.ndarray)):  
                prediction_value = prediction[0]  # Extrai o primeiro valor
            else:
                prediction_value = prediction  # Caso já seja um número
            
            # Gera a recomendação com base na previsão
            recommendation = generate_recommendation(prediction_value)
            
            print("Prediction:", prediction_value, "Recommendation:", recommendation)  # Log do resultado
            
            return jsonify({
                'prediction': prediction_value,  # Deve ser int
                'recommendation': recommendation
            })
        
        except Exception as e:
            print("Error in prediction route:", str(e))  # Log do erro
            return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

