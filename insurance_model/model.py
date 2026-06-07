from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import math
import numpy as np
import matplotlib.pyplot as plt
from snowflake_info import df
df = df.reset_index()
df = df.rename(columns={'index': 'id'})
x = np.array(df[['AGE','SEX','BMI','CHILDREN','SMOKER','NORTHEAST','NORTHWEST','SOUTHEAST','SOUTHWEST']]).reshape(-1, 9)
y = np.array(df['CHARGES']).reshape(-1, 1)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

model = LinearRegression()
model.fit(x_train_poly, y_train)

y_pred = model.predict(x_test_poly)

plt.scatter(y_test, y_pred, color='red', alpha=0.6)


min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')


plt.savefig('actual_vs_predicted.png')

print("Mean Squared Error:", math.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2 Score:", r2_score(y_test, y_pred))

# input_data = np.array([[21, 1, 34.6, 0, 0, 0, 0, 0, 1]])
# input_data_poly = poly.transform(input_data)
# predicted_charge = model.predict(input_data_poly)
# print("Predicted Insurance Charge:", predicted_charge[0][0])

