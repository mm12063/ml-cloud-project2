from flask import Flask, render_template, request, jsonify
import Predictor

app = Flask(__name__,template_folder='static/templates')

TICKERS = {
    "AXP": "American Express Co",
    "AMGN": "Amgen Inc",
    "AAPL": "Apple Inc",
    "BA": "Boeing Co",
    "CAT": "Caterpillar Inc",
    "CSCO": "Cisco Systems Inc",
    "CVX": "Chevron Corp",
    "GS": "Goldman Sachs Group Inc",
    "HD": "Home Depot Inc",
    "HON": "Honeywell International Inc",
    "IBM": "IBM Corp",
    "INTC": "Intel Corp",
    "JNJ": "Johnson & Johnson",
    "KO": "Coca-Cola Co",
    "JPM": "JPMorgan Chase & Co",
    "MCD": "McDonald’s Corp",
    "MMM": "3M Co",
    "MRK": "Merck & Co Inc",
    "MSFT": "Microsoft Corp",
    "NKE": "Nike Inc",
    "PG": "Procter & Gamble Co",
    "TRV": "Travelers Companies Inc",
    "UNH": "UnitedHealth Group Inc",
    "CRM": "Salesforce Inc",
    "VZ": "Verizon Communications Inc",
    "V": "Visa Inc",
    "WBA": "Walgreens Boots Alliance Inc",
    "WMT": "Walmart Inc",
    "DIS": "Walt Disney Co",
    "DOW": "Dow Inc"
}

@app.route("/")
def index():
    return render_template("index.html", tickers=TICKERS)


@app.route("/prediction", methods=["GET"])
def get_prediction():
    ticker = request.args.get("ticker")
    result = 'Ticker not passed to get_predictor()'
    if ticker:
        result = Predictor.get_ticker_prediction(ticker)

    return jsonify(result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
