"""
education.py
Plain-language educational content for stock market beginners.
No emojis. Professional tone.
"""

GLOSSARY = {
    "Stock": (
        "A stock is a small piece of ownership in a company. When you buy a stock, "
        "you become a part-owner (shareholder) of that company. If the company grows "
        "and earns more profit, your stock becomes more valuable."
    ),
    "Share Price": (
        "The price at which one unit of a company's stock is currently being bought "
        "or sold in the market. Prices change throughout the trading day based on "
        "supply and demand."
    ),
    "NSE / BSE": (
        "NSE (National Stock Exchange) and BSE (Bombay Stock Exchange) are the two "
        "major stock exchanges in India where company stocks are listed and traded. "
        "NSE is the larger of the two by trading volume."
    ),
    "NIFTY 50": (
        "NIFTY 50 is an index that tracks the performance of the 50 largest and most "
        "liquid companies listed on NSE. It is used as a benchmark to measure the "
        "overall health of the Indian stock market."
    ),
    "SENSEX": (
        "SENSEX is a similar index for BSE, tracking the 30 largest companies. "
        "When people say 'the market went up today', they often refer to movement "
        "in NIFTY or SENSEX."
    ),
    "RSI (Relative Strength Index)": (
        "RSI is a number between 0 and 100. A reading above 70 suggests a stock "
        "has been rising fast and may slow down (overbought). Below 30 suggests "
        "it has been falling fast and may recover (oversold). It helps identify "
        "potential turning points."
    ),
    "MACD": (
        "MACD (Moving Average Convergence Divergence) is an indicator that shows the "
        "relationship between two moving averages of a stock's price. When the MACD "
        "line crosses above its signal line, it is considered a bullish (positive) "
        "sign. When it crosses below, it is bearish (negative)."
    ),
    "EMA (Exponential Moving Average)": (
        "EMA is the average price of a stock over a given period (e.g. 20 days), "
        "with more weight given to recent prices. Traders use EMA to identify the "
        "direction of the trend. If the current price is above the 50-day EMA, "
        "the stock is generally considered to be in an uptrend."
    ),
    "Bollinger Bands": (
        "Bollinger Bands are three lines plotted around a stock's price: a middle "
        "line (average) and two outer bands. When the price touches the upper band, "
        "the stock may be overbought. When it touches the lower band, it may be "
        "oversold. Wide bands indicate high volatility; narrow bands indicate "
        "a quiet period."
    ),
    "Volatility": (
        "Volatility measures how much a stock's price swings up and down. A highly "
        "volatile stock can give large gains but also large losses. Beginners are "
        "generally advised to start with less volatile, well-established companies."
    ),
    "P&L (Profit and Loss)": (
        "P&L is the difference between what you paid for a stock and what it is "
        "currently worth. Positive P&L means you are in profit; negative means "
        "you are currently at a loss. Losses are only realized when you actually sell."
    ),
    "Dividend": (
        "Some companies share a portion of their profits with shareholders in the "
        "form of dividends - a cash payment made periodically. Dividend-paying stocks "
        "provide a regular income in addition to any price appreciation."
    ),
    "Market Capitalization": (
        "Market cap is the total value of all shares of a company. Large-cap companies "
        "(like Reliance or TCS) are big, established, and generally more stable. "
        "Small-cap companies are smaller and can grow faster but carry more risk."
    ),
    "Bull Market": (
        "A bull market is a period when stock prices are generally rising. Investor "
        "confidence is high, the economy is usually growing, and it is considered a "
        "good time to be invested in the market."
    ),
    "Bear Market": (
        "A bear market is a period when stock prices fall significantly (usually 20% "
        "or more from a recent high). It often reflects economic slowdown or "
        "uncertainty. Bear markets are part of the normal market cycle."
    ),
    "Intraday Trading": (
        "Intraday trading means buying and selling stocks within the same trading day. "
        "All positions are closed before the market closes at 3:30 PM IST. This "
        "carries high risk and requires experience. Beginners are advised to avoid it."
    ),
    "Fundamental Analysis": (
        "Evaluating a company by looking at its financial health - revenues, profits, "
        "debt, and business model - to determine whether its stock is fairly priced. "
        "This is a long-term approach."
    ),
    "Technical Analysis": (
        "Studying historical price charts and statistical indicators (like RSI, MACD) "
        "to forecast future price movements. This tool uses technical analysis to "
        "generate predictions."
    ),
    "SHAP Values": (
        "SHAP (SHapley Additive exPlanations) is a technique that explains why our "
        "machine learning model made a specific prediction. It shows which factors "
        "(indicators) pushed the prediction toward 'UP' or 'DOWN' and by how much. "
        "Higher SHAP value for a feature means it had a stronger influence."
    ),
    "LIME Explanation": (
        "LIME (Local Interpretable Model-agnostic Explanations) is another method to "
        "explain individual predictions. It shows which features (like RSI or MACD) "
        "supported or opposed the model's prediction for a specific stock at a "
        "specific moment."
    ),
}

BEGINNER_TIPS = [
    "Never invest money you cannot afford to lose. Start with a small amount "
    "and learn as you go.",
    "Diversify your investments across different sectors (IT, Banking, Pharma, etc.) "
    "to reduce risk. If one sector falls, others may hold steady.",
    "Invest in companies whose business you understand. If you do not understand "
    "what a company does, it is harder to judge its prospects.",
    "The stock market has historically risen over long periods. Patience and "
    "staying invested through short-term volatility is key.",
    "Avoid making decisions based on rumors or social media tips. Always check "
    "fundamentals and reliable data.",
    "A stock price going down does not automatically mean you should sell. "
    "Evaluate whether the company's business has actually deteriorated.",
    "Past performance of a stock does not guarantee future results. Predictions "
    "shown here are based on patterns, not certainties.",
    "Consider investing through a Systematic Investment Plan (SIP) in index funds "
    "or mutual funds if you are a beginner - it reduces timing risk.",
    "Track your investments regularly but avoid checking prices every hour. "
    "Emotional reactions to daily fluctuations lead to poor decisions.",
    "Consult a SEBI-registered investment advisor for personalized financial advice. "
    "This tool is for educational and informational purposes only.",
]

HOW_TO_READ_PREDICTION = """
How to Interpret the AI Prediction
-----------------------------------
The model predicts whether a stock's price is likely to go UP or DOWN on the next trading day.

- Direction: UP means the model expects the closing price tomorrow to be higher than today.
- Confidence: A higher percentage means the model is more certain. 
  Anything below 60% should be treated with caution.
- SHAP Chart: Shows which indicators (features) most influenced the overall model behavior.
  Features with larger bars had a stronger effect on predictions.
- LIME Chart: Shows why the model predicted what it did for THIS specific stock RIGHT NOW.
  Bars pointing right (positive) supported the UP prediction; bars pointing left opposed it.

IMPORTANT: No model can predict the market with 100% accuracy. Use this as one of several 
inputs to your research, not as the sole basis for any investment decision.
"""

SECTOR_EXPLAINERS = {
    "Banking": "Banks earn income from lending money and providing financial services. "
               "Their performance is closely tied to interest rates set by the RBI and "
               "the overall economic environment. Strong GDP growth usually benefits banks.",
    "IT": "Indian IT companies earn a large portion of revenue in US Dollars by providing "
          "software services to global clients. They tend to do well when global tech "
          "spending is high. Currency movements (USD vs INR) also affect their profits.",
    "FMCG": "Fast Moving Consumer Goods companies sell everyday products (soap, biscuits, "
            "tea). They are considered defensive stocks because people buy these products "
            "regardless of economic conditions, making them relatively stable investments.",
    "Pharma": "Pharmaceutical companies manufacture medicines and healthcare products. "
              "Indian pharma exports heavily to the US and Europe. Regulatory approvals "
              "from bodies like the US FDA can significantly impact their stock price.",
    "Energy": "Energy stocks include oil & gas and power companies. Their performance is "
              "linked to global crude oil prices and domestic energy demand. They often pay "
              "steady dividends.",
    "Auto": "Automobile companies are sensitive to economic cycles, fuel prices, interest "
            "rates (which affect car loan costs), and new technology trends like electric "
            "vehicles. Consumer sentiment plays a big role.",
    "Telecom": "Telecom companies provide mobile and internet services. This is a "
               "capital-intensive sector. ARPU (Average Revenue Per User) is a key metric "
               "investors watch.",
}
