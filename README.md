# 🧴 מסווג שגרת טיפוח

פרויקט למידת מכונה שמסווג את שגרת הטיפוח המתאימה ביותר (Routine_A עד Routine_D) לפי מאפיינים אישיים כמו סוג עור, שעות שינה, שתיית מים, חשיפה לשמש ועוד.

---

## 🎯 מטרת הפרויקט

- לבנות מודל חכם שממליץ על שגרת טיפוח מותאמת אישית.
- להשוות בין מודלים שונים לבעיה של סיווג רב-קטגורי.
- להציג את ביצועי המודלים באמצעות גרפים ומדדים.

---

## 📊 סקירת הנתונים

העמודות בקובץ כוללות:

- `Age` – גיל
- `Skin_Type` – סוג עור (שומני, יבש, רגיל...)
- `Sun_Exposure` – חשיפה לשמש (Low, Medium, High)
- `Uses_Products` – שימוש במוצרי טיפוח
- `Uses_Makeup` – שימוש באיפור (True/False)
- `Sleep_Hours` – שעות שינה
- `Water_Intake` – שתיית מים ביום
- `Screen_Time_Hours` – זמן מסך יומי
- `Recommended_Routine` – תווית מטרה (שגרת טיפוח)

---

## ⚙️ שלבי העבודה

1. **טעינת הנתונים** – שימוש בקובץ Excel עם `pandas`.
2. **חקירה ראשונית** – הצגת מבנה הטבלה וסטטיסטיקות בסיסיות.
3. **עיבוד נתונים**:
   - קידוד משתנים קטגוריים (Label/One-Hot)
   - נרמול ערכים מספריים עם `StandardScaler`
4. **פיצול לסטים** – `train_test_split` עם stratify
5. **אימון מודלים**:
   - Logistic Regression
   - Random Forest
   - K-Nearest Neighbors (k=10)
6. **חיזוי והערכת ביצועים** – מטריצת בלבול, דיוק, ודוחות סיווג
7. **השוואה ויזואלית** – גרפים של דיוק, פיזור תחזיות ועוד
8. **בחירת מודל סופי** – לפי ביצועים

## 🖼 תוצרים גרפיים

- מטריצות בלבול עבור כל מודל
- גרפי scatter – ניבוי מול אמת. הערה: בחרתי בעיית סיווג, ולכן הפלט אינו רציף – מה שמוביל לגרף פחות רציף.
- גרף השוואת דיוקים (bar chart)

---

## 🚀 כיצד להריץ

1. ודאו שהתקנתם את הספריות הדרושות:
   ```bash
   pip install pandas scikit-learn matplotlib seaborn openpyxl
