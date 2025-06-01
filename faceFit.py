import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import numpy as np

#בחרתי בעיית סיווג, ולכן הפלט אינו רציף – מה שמוביל לגרף פחות רציף.

#1
#בחירת Data Frame
#  שינוי ברירת מחדל-להראות את כל העמודות
pd.set_option('display.max_columns', None)
# טוענים את הקובץ
df = pd.read_excel("skin_routine_data.xlsx")

#2
# טעינת נתונים וחקירה
print("חמש השורות הראשונות:")
print(df.head())

print("\nמידע כללי על הנתונים:")
print(df.info())

print("\nתיאור סטטיסטי של העמודות המספריות:")
print(df.describe())

#3
#עיבוד נתונים מראש
#קידוד עמודות קטגוריות לא סדורות
# ביצוע One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Skin_Type', 'Uses_Products'],dtype=int)

#קידוד עמודות סדורות
df_encoded['Sun_Exposure'] = df_encoded['Sun_Exposure'].map({'Low': 0, 'Medium': 1, 'High': 2})

#קידוד בוליאני
df_encoded['Uses_Makeup'] = df_encoded['Uses_Makeup'].map({True: 1, False: 0})

#קידוד תכונת מטרה
le = LabelEncoder()
df_encoded['Recommended_Routine'] = le.fit_transform(df_encoded['Recommended_Routine'])
print("\nהעמודות אחרי One-Hot Encoding:")
print(df_encoded.head())

#נרמול
#נשים רק עמודות מספריות ולא עמודות מקודדות
scaler = StandardScaler()
numeric_cols = ["Age","Sleep_Hours","Water_Intake","Screen_Time_Hours"]
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
print("\nהעמודות בטבלה לאחר הנרמול:")
print(df_encoded[numeric_cols])
print("\nכל העמודות בטבלה לאחר הנרמול:")
print(df_encoded.head())

#4
# בחירת תכונת יציאה
y = df_encoded['Recommended_Routine']
X = df_encoded.drop('Recommended_Routine', axis=1)

#5
#מחלק את הנתונים ל־4 משתנים- ללימוד ובדיקה
#stratify=y מוודא שכל הקטגוריות של הפלט יהיו גם בלימוד וגם בבדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

#בדיקה כמה כל קטגוריה נמצאת בדאטה וכמה בבדיקה
print(y.value_counts())
print(y_test.value_counts())

#יוצרים אובייקט של מודל Logistic Regression
model = LogisticRegression(max_iter=1000)

#אימון המודל
model.fit(X_train, y_train)

#חיזוי על נתוני הבדיקה
y_pred = model.predict(X_test)

#מחשבים את הדיוק (Accuracy) — אחוז התחזיות שהמודל ניבא נכון מתוך כל הדוגמאות בסט הבדיקה.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy על סט הבדיקה: {accuracy:.4f}")

print("\nדוח סיווג מפורט:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 6
# מודל סופי ותחזיות
# מאמנים את המודל על כל הנתונים הזמינים (לימוד + בדיקה)
model_final = LogisticRegression(max_iter=1000)
model_final.fit(X, y)


#7
# מודל חדש - Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"\nAccuracy של Random Forest על סט הבדיקה: {accuracy_rf:.4f}")

print("\nדוח סיווג מפורט ל-Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# השוואה בין Logistic Regression ל-Random Forest
print(f"\nAccuracy של Logistic Regression: {accuracy:.4f}")
print(f"Accuracy של Random Forest: {accuracy_rf:.4f}")


# מודל חדש - K-Nearest Neighbors
model_knn = KNeighborsClassifier(n_neighbors=10)
model_knn.fit(X_train, y_train)

# חיזוי עם KNN
y_pred_knn = model_knn.predict(X_test)

# דיוק
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"\nAccuracy של KNN על סט הבדיקה: {accuracy_knn:.4f}")

# דוח סיווג
print("\nדוח סיווג מפורט ל-KNN:")
print(classification_report(y_test, y_pred_knn, target_names=le.classes_))

# השוואה מול המודלים
print(f"\nהשוואה בין המודלים:")
print(f"Accuracy של Logistic Regression: {accuracy:.4f}")
print(f"Accuracy של KNN: {accuracy_knn:.4f}")
print(f"Accuracy של Random Forest: {accuracy_rf:.4f}")

#8
#  הצגת מסקנות ומדדים ויזואליים

# פונקציה להצגת Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()

# הצגת Confusion Matrix לכל מודל
plot_confusion_matrix(y_test, y_pred, le.classes_, "Confusion Matrix - Logistic Regression")
plot_confusion_matrix(y_test, y_pred_rf, le.classes_, "Confusion Matrix - Random Forest")
plot_confusion_matrix(y_test, y_pred_knn, le.classes_, "Confusion Matrix - KNN")

# גרף Accuracy של כל המודלים
models = ['Logistic Regression', 'Random Forest', 'KNN']
accuracies = [accuracy, accuracy_rf, accuracy_knn]

plt.figure(figsize=(8,5))
sns.barplot(x=models, y=accuracies)
plt.ylim(0,1)
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison Between Models')
plt.show()


#גרף פיזור בין התחזיות לערכים האמיתיים עבור כל מודל
def plot_scatter_with_regression(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_true, color='blue', label='Points')

    # התאמת קו רגרסיה (רק לצורך המחשה)
    reg = LinearRegression().fit(np.array(y_pred).reshape(-1, 1), y_true)
    line_x = np.linspace(min(y_pred), max(y_pred), 100)
    line_y = reg.predict(line_x.reshape(-1, 1))
    plt.plot(line_x, line_y, color='red', label='Regression Line')

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} - Scatter Plot with Regression Line')
    plt.legend()
    plt.grid(True)
    plt.show()

# קריאה לפונקציה עבור כל מודל
plot_scatter_with_regression(y_test, y_pred, "Logistic Regression")
plot_scatter_with_regression(y_test, y_pred_rf, "Random Forest")
plot_scatter_with_regression(y_test, y_pred_knn, "K-Nearest Neighbors")

#9
#הדפסה של מסקנתך האישית: באיזה מודל מומלץ להשתמש?
print("\nמסקנה:")
print("Random Forest ו Logistic Regression הם המומלצים ביותר")


