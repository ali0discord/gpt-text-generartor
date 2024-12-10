import sqlite3

def create_db():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS inputs (
            id INTEGER PRIMARY KEY,
            input_text TEXT,
            selected_model TEXT,
            seed INTEGER
        )
    """)
    conn.commit()
    conn.close()

def insert_into_db(input_text, selected_model, seed):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO inputs (input_text, selected_model, seed)
        VALUES (?, ?, ?)
    """, (input_text, selected_model, seed))
    conn.commit()
    conn.close()

def clear_database():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("DELETE FROM inputs")
    conn.commit()
    conn.close()
