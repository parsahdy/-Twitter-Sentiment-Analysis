from inference import predicte_sentiment
import pandas as pd

test = "I absolutly love this movie!!."
df = pd.DataFrame({"Clean_Text": [test]})
result = predicte_sentiment(df)

print(result[["Clean_Text", "prediction", "label"]])