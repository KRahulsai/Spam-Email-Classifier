import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

msg = input("Enter message: ")
data = vectorizer.transform([msg])

prediction = model.predict(data)

print("Spam" if prediction[0] == 1 else "Not Spam")
