import sqlite3

database_path = 'database.db'


def create_db():
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS inputs (
            id INTEGER PRIMARY KEY,
            input_text TEXT,
            selected_model TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_into_db(input_text, selected_model):
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute("""
        INSERT INTO inputs (input_text, selected_model)
        VALUES (?, ?)
    """, (input_text, selected_model))
    conn.commit()
    conn.close()

def clear_database():
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    c.execute("DELETE FROM inputs")
    conn.commit()
    conn.close()

def fetch_all_inputs():
    """
    بازیابی تمام ورودی‌ها از دیتابیس.
    """
    try:
        conn = sqlite3.connect(database_path)
        c = conn.cursor()

        # فرض کنید دیتابیس جدولی به نام 'inputs' با ستون‌های input_text و model_name دارد
        c.execute("SELECT input_text, model_name FROM inputs")
        results = c.fetchall()
        conn.close()
        return results
    except sqlite3.Error as e:
        print(f"Error fetching inputs from database: {e}")
        return []
