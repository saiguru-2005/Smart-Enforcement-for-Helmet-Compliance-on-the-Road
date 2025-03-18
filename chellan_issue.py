import sqlite3

def issue_challan(license_plate):
    license_plate= license_plate.replace(" ","")
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS riders (LicensePlate TEXT PRIMARY KEY, ChellanAmount INTEGER, Reason TEXT, ChellanCount INTEGER DEFAULT 0)")

    # Check if the rider exists in the database
    cursor.execute("SELECT * FROM riders WHERE LicensePlate = ?", (license_plate,))
    rider = cursor.fetchone()

    if rider:
        chellan_count = rider[3] + 1  
        cursor.execute("UPDATE riders SET ChellanAmount = ?, Reason = ?, ChellanCount = ? WHERE LicensePlate = ?",
                       (rider[1] + 200, "No helmet", chellan_count, license_plate))
        conn.commit()
        print("Additional Chellan issued successfully for license plate:", license_plate)
    else:
        # Rider does not exist, insert rider details and issue a new Chellan
        cursor.execute("INSERT INTO riders (LicensePlate, ChellanAmount, Reason, ChellanCount) VALUES (?, ?, ?, ?)",
                       (license_plate, 200, "No helmet", 1))
        conn.commit()
        print("New rider details added and Chellan issued successfully for license plate:", license_plate)

    conn.close()

# issue_chellan("ABC123")  