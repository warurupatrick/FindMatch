from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import similarity_search
import upload_to_pinecone
import os
import base64

app = Flask(__name__)
CORS(app)

# Absolute paths for log and results
LOG_PATH = r"C:\Users\REGINAH\Desktop\ml\log.txt"
#RESULTS_FOLDER = r"C:\Users\PATO\OneDrive\Desktop\ml\results"
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

import base64

@app.route('/start-matching', methods=['POST'])
def start_matching():
    try:
        print("Starting Upload to Pinecone...")
        upload_to_pinecone.main()

        print("\nRunning Similarity Search...")
        similarity_search.main()

        # Read log file
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as file:
                log_contents = file.read()
        else:
            log_contents = "Log file not found."

        # List image results
        image_files = [f for f in os.listdir(RESULTS_FOLDER)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_urls = [f"/results/{filename}" for filename in image_files]

        return jsonify({
            'message': 'Matching completed successfully.',
            'log': log_contents,
            'images': image_urls
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/base64')

def serve_images_base64():
    try:
        image_data = []

        for filename in os.listdir(RESULTS_FOLDER):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                file_path = os.path.join(RESULTS_FOLDER, filename)

                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

                image_data.append({
                    'filename': filename,
                    'base64': encoded_string
                })

        return jsonify({'images': image_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

import re
from flask import jsonify

@app.route('/results/log-summary', methods=['GET'])
def summarize_log():
    log_path = r"C:\Users\REGINAH\Desktop\ml\log.txt"

    try:
        matches = []
        with open(log_path, 'r') as file:
            for line in file:
                match = re.search(r'Match \d+: ID: ([^|]+)\s+\|\s+Score: ([0-9.]+)', line)
                if match:
                    match_id = match.group(1).strip()
                    score = float(match.group(2))
                    matches.append({'id': match_id, 'score': score})

        return jsonify({'matches': matches}), 200

    except FileNotFoundError:
        return jsonify({'error': 'log.txt not found'}), 404

    except Exception as e:
        return jsonify({'error': str(e)}), 500
import json
import requests
@app.route('/send-message', methods=['POST'])
def send_message():
    data = request.get_json()
    url = 'https://portal.zettatel.com/SMSApi/send'

    telephone = data.get('telephone')
    message = data.get('message')

    if not telephone or not message:
        return jsonify({'error': 'Both telephone and message are required'}), 400

    # Format the telephone number
    tel_no = "254{}".format(telephone)


    try:
        payload = {
            "userid": "Willy",
            "password": "GX98BFfh",
            "senderid": "FINDMATCH",
            "msgType": "text",
            "duplicatecheck": "true",
            "sendMethod": "quick",
            "sms": [
                {
                    "mobile": [tel_no],
                    "msg": message
                }
            ]
        }

        json_payload = json.dumps(payload)
        headers = {'Content-Type': 'application/json'}

        response = requests.post(url, headers=headers, data=json_payload)
        print("Message sent. Response:", response.text)

    except Exception as e:
        print("Error sending message:", str(e))
        return jsonify({'error': 'Failed to send message'}), 500

    print(f"Received message from {telephone}: {message}")
    return jsonify({'message': 'Message received successfully'}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)











# from flask import Flask, jsonify, request
# from flask_cors import CORS

# import similarity_search
# import upload_to_pinecone

# app = Flask(__name__)
# CORS(app)

# @app.route('/start-matching', methods=['POST'])
# def start_matching():
#     try:
#         print("Starting Upload to Pinecone...")
#         upload_to_pinecone.main()  # Make sure upload_to_pinecone.py has a main() function

#         print("\nRunning Similarity Search...")
#         similarity_search.main()  # Make sure similarity_search.py has a main() function

#         return jsonify({'message': 'Matching completed successfully.'}), 200

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=7000)
