import pandas as pd

# Create a DataFrame with the provided data
data = {
    # ["delivery_id", "city", "courier", "age", "experience", "distance_km", "delivery_time_min", "rating", "weather", "price"],
    # [1, "Casablanca", "Ahmed", 25, 2, 5.2, 28, 4.5, "Sunny", 35],
    # [2, "Rabat", "Youssef", 31, 5, 8.1, 42, 4.2, "Rain", 50],
    # [3, "Casablanca", "Hamza", 22, 1, 3.5, 20, 4.8, "Sunny", 25],
    # [4, "Marrakech", "Omar", 28, 4, 12.0, 55, 3.9, "Windy", 70],
    # [5, "Rabat", "Ayoub", 35, 8, 6.7, 35, 4.6, "Sunny", 45],
    # [6, "Agadir", "Mehdi", 24, 2, 9.5, 48, 4.0, "Rain", 60],
    # [7, "Casablanca", "Karim", 29, 6, 15.2, 65, 3.7, "Rain", 85],
    # [8, "Marrakech", "Anas", 26, 3, 4.8, 30, 4.4, "Sunny", 40],
    # [9, "Agadir", "Reda", 32, 7, 7.3, 38, 4.7, "Windy", 55],
    # [10, "Rabat", "Ismail", 23, 1, 11.5, 52, 3.8, "Rain", 65],
    # [11, "Casablanca", "Bilal", 27, 4, 6.2, 33, 4.3, "Sunny", 42],
    # [12, "Agadir", "Zakaria", 30, 5, 13.8, 58, 4.1, "Windy", 75],
    # [13, "Marrakech", "Soufiane", 21, 1, 2.9, 18, 4.9, "Sunny", 20],
    # [14, "Rabat", "Hamid", 38, 10, 5.5, 31, 4.5, "Sunny", 38],
    # [15, "Casablanca", "Othmane", 33, 7, 10.4, 47, 4.0, "Windy", 62]
    'delivery_id' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'city' : ["Casablanca", "Rabat", "Casablanca", "Marrakech", "Rabat", "Agadir", "Casablanca", "Marrakech", "Agadir", "Rabat", "Casablanca", "Agadir", "Marrakech", "Rabat", "Casablanca"],
    'courier' : ["Ahmed", "Youssef", "Hamza", "Omar", "Ayoub", "Mehdi", "Karim", "Anas", "Reda", "Ismail", "Bilal", "Zakaria", "Soufiane", "Hamid", "Othmane"],
    'age' : [25, 2, 22, 28, 35, 24, 29, 26, 32, 23, 27, 30, 21, 38, 33],
    'experience' : [2, 5, 1, 4, 8, 2, 6, 3, 7, 1, 4, 5, 1, 10, 7],
    'distance_km' : [5.2, 8.1, 3.5, 12.0, 6.7, 9.5, 15.2, 4.8, 7.3, 11.5, 6.2, 13.8, 2.9, 5.5, 10.4],
    'delivery_time_min' : [28, 42, 20, 55, 35, 48, 65, 30, 38, 52, 33, 58, 18, 31, 47],
    'rating' : [4.5, 4.2, 4.8, 3.9, 4.6, 4.0, 3.7, 4.4, 4.7, 3.8, 4.3, 4.1, 4.9, 4.5, 4.0],
    'weather' : ["Sunny", "Rain", "Sunny", "Windy", "Sunny", "Rain", "Rain", "Sunny", "Windy", "Rain", "Sunny", "Windy", "Sunny", "Sunny", "Windy"],
    'price' : [35, 50, 25, 70, 45, 60, 85, 40, 55, 65, 42, 75, 20, 38, 62]
}


df = pd.DataFrame(data)

# print(df.head())
# print("*************************************")
# print(df.tail())
# print("*************************************")
# print(df.shape)
# print("*************************************")
# print(df.columns)

# print(df.info())
# print(df.describe())
# print(df.dtypes)


clos = {
    'age' ,
    'experience',
    'distance_km',
    'delivery_time_min'

}

liste = []
for i in clos :
    liste.append((i , df[df[i] > 10][i].count()))

mainCity = df.groupby('city')['age'].mean()



print(mainCity)