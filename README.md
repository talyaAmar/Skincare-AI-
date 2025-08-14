# 🧴 Skincare Routine Classifier Project

A machine learning project that classifies the most suitable skincare routine (Routine_A to Routine_D) based on personal characteristics such as skin type, sleep hours, water intake, sun exposure, and more.

---

## 🎯 Project Goal

* To build a smart model that recommends a personalized skincare routine.
* To compare different models for a multi-category classification problem.
* To present the performance of the models using graphs and metrics.

---

## 📊 Data Overview

The columns in the file include:

* Age
* Skin_Type (Oily, Dry, Normal...)
* Sun_Exposure (Low, Medium, High)
* Uses_Products
* Uses_Makeup (True/False)
* Sleep_Hours
* Water_Intake (daily)
* Screen_Time_Hours (daily)
* Recommended_Routine (target label)

---

## ⚙️ Work Stages

1.  **Loading the data** – Using an Excel file with pandas.
2.  **Initial exploration** – Displaying the table structure and basic statistics.
3.  **Data processing**:
    - Encoding categorical variables (Label/One-Hot)
    - Normalizing numerical values with StandardScaler
4.  **Splitting into sets** – train_test_split with stratify
5.  **Training models**:
    - Logistic Regression
    - Random Forest
    - K-Nearest Neighbors (k=10)
6.  **Prediction and performance evaluation** – Confusion matrix, accuracy, and classification reports
7.  **Visual comparison** – Graphs of accuracy, prediction scatter, etc.
8.  **Choosing the final model** – Based on performance

## 🖼 Graphical Outputs

* Confusion matrices for each model
* Scatter plots – prediction vs. true. Note: I chose a classification problem, so the output is not continuous – leading to a less continuous graph.
* Accuracy comparison graph (bar chart)

---

## 🚀 How to Run

 Make sure you have installed the necessary libraries:
   
```bash
    pip install pandas scikit-learn matplotlib seaborn openpyxl
```
   
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 🧴 מסווג שגרת טיפוח

פרויקט למידת מכונה שמסווג את שגרת הטיפוח המתאימה ביותר (Routine_A עד Routine_D) לפי מאפיינים אישיים כמו סוג עור, שעות שינה, שתיית מים, חשיפה לשמש ועוד.

---

## 🎯 מטרת הפרויקט

* לבנות מודל חכם שממליץ על שגרת טיפוח מותאמת אישית.
* להשוות בין מודלים שונים לבעיה של סיווג רב-קטגורי.
* להציג את ביצועי המודלים באמצעות גרפים ומדדים.

---

## 📊 סקירת הנתונים

העמודות בקובץ כוללות:

* Age – גיל
* Skin_Type – סוג עור (שומני, יבש, רגיל...)
* Sun_Exposure – חשיפה לשמש (Low, Medium, High)
* Uses_Products – שימוש במוצרי טיפוח
* Uses_Makeup – שימוש באיפור (True/False)
* Sleep_Hours – שעות שינה
* Water_Intake – שתיית מים ביום
* Screen_Time_Hours – זמן מסך יומי
* Recommended_Routine – תווית מטרה (שגרת טיפוח)

---

## ⚙️ שלבי העבודה

1. **טעינת הנתונים** – שימוש בקובץ Excel עם pandas.
2. **חקירה ראשונית** – הצגת מבנה הטבלה וסטטיסטיקות בסיסיות.
3. **עיבוד נתונים**:
   - קידוד משתנים קטגוריים (Label/One-Hot)
   - נרמול ערכים מספריים עם StandardScaler
4. **פיצול לסטים** – train_test_split עם stratify
5. **אימון מודלים**:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors (k=10)
6. **חיזוי והערכת ביצועים** – מטריצת בלבול, דיוק, ודוחות סיווג
7. **השוואה ויזואלית** – גרפים של דיוק, פיזור תחזיות ועוד
8. **בחירת מודל סופי** – לפי ביצועים

## 🖼 תוצרים גרפיים

* מטריצות בלבול עבור כל מודל
* גרפי scatter – ניבוי מול אמת. הערה: בחרתי בעיית סיווג, ולכן הפלט אינו רציף – מה שמוביל לגרף פחות רציף.
* גרף השוואת דיוקים (bar chart)

---

## 🚀 כיצד להריץ

ודאו שהתקנתם את הספריות הדרושות:
   
```bash
   pip install pandas scikit-learn matplotlib seaborn openpyxl
```
