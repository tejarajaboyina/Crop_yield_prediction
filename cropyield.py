import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=pd.read_csv("crop_yield.csv")
state=input("Enter state")
crop_name=input("Enter crop")
year=int(input("Enter year"))
filtered=df[(df["State"]==state)&(df["Crop"]==crop_name)]
filtered=filtered.sort_values(by="Crop_Year")
X=filtered["Crop_Year"].values.reshape(-1,1)
y=filtered["Yield"].values
model=LinearRegression()
model.fit(X,y)
predicated_yield=model.predict([[year]])[0]
last_year = filtered["Crop_Year"].max()
last_yield = filtered.loc[filtered["Crop_Year"] == last_year, "Yield"].values[0]
diff = predicated_yield - last_yield
percent_change = (diff / last_yield) * 100
if diff > 0:
    status = "increase"
else:
    status = "decrease"

print(f"In {year}, {state} ({crop_name}) yield ≈ {predicated_yield:.2f} t/ha,")
print(f"which is a {abs(percent_change):.2f}% {status} compared to {last_year}.")
plt.plot(filtered["Crop_Year"], filtered["Yield"], label="Historical Yield")
plt.scatter(year, predicated_yield, color="red", label=f"Predicted Yield ({year})")
plt.xlabel("Crop Year")
plt.ylabel("Yield (t/ha)")
plt.title(f"Yield Prediction for {crop_name} in {state}")
plt.legend()
plt.grid(True)
plt.show()



