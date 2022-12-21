from flask import Flask, jsonify, request
from flask_cors import CORS
import inference

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return "Please pass a ticker symbol to the API: /inference/symb"


@app.route("/inference", methods=["GET"])
def get_prediction():
    symb = request.args.get("ticker")
    return_dictionary = {}
    if symb:
        train_rmse = inference.get_rmse(symb)
        testScore, y_test, y_test_pred, dates = inference.get_inference(symb)
        return_dictionary = {"train_rmse": train_rmse,
                             "test_rmse": testScore,
                             "y_test": y_test.tolist(),
                             "y_test_pred": y_test_pred.tolist(),
                             "dates": dates}

    return jsonify(return_dictionary)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)


