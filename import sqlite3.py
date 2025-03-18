import sqlite3

# Connect to the database
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

# Step 1: Check if the "license_plate" column exists
cursor.execute("PRAGMA table_info(riders);")
columns = [col[1] for col in cursor.fetchall()]  # Extract column names

if "license_plate" not in columns:
    print("Column 'license_plate' is missing. Adding it now...")
    cursor.execute("ALTER TABLE riders ADD COLUMN license_plate TEXT;")
    conn.commit()
else:
    print("Column 'license_plate' exists.")

# Step 2: Check if the column has data
cursor.execute("SELECT license_plate FROM riders LIMIT 5;")
data = cursor.fetchall()

if not data or all(value[0] is None for value in data):
    print("Column 'license_plate' has no data. Updating it from 'number_plate'...")
    cursor.execute("UPDATE riders SET license_plate = number_plate;")
    conn.commit()
else:
    print("Column 'license_plate' already contains data.")

# Step 3: Drop the "number_plate" column (SQLite does not support direct column drop)
if "number_plate" in columns:
    print("Removing old 'number_plate' column...")
    cursor.execute("""
        CREATE TABLE riders_new AS SELECT id, name, license_plate FROM riders;
    """)  # Recreate the table with only required columns
    cursor.execute("DROP TABLE riders;")
    cursor.execute("ALTER TABLE riders_new RENAME TO riders;")
    conn.commit()

print("Database update complete!")

# Close the connection
conn.close()
