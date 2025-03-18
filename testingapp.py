
#from flask import Flask, render_template, request
#import sqlite3

#app = Flask(__name__)

# # Function to query the database for chellans issued to a rider
# def get_chellans(number_plate):
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()
#     # cursor.execute("SELECT issued_chellan FROM riders WHERE number_plate=?", (number_plate,))
#     # chellans = cursor.fetchone()
#     # conn.close()
#     # return chellans[0] if chellans else 0
#     query = f"SELECT issued_chellan FROM riders WHERE number_plate = '{number_plate}';"
#     cursor.execute(query)
#     result = cursor.fetchone()
#     print (result)
#     conn.close()
#     if result is not None:
#         return result[0]
#     else:
#         return 'no chellans'

# # Route for the home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Route for handling form submission
# @app.route('/check_chellans', methods=['POST'])
# def check_chellans():
#     if request.method == 'POST':
#         number_plate = request.form['number_plate']
#         chellans = get_chellans(number_plate)
#         return render_template('result.html', number_plate=number_plate, count=chellans)

# if __name__ == '__main__':
#     app.run(debug=True)


#second version is from below

# from flask import Flask, render_template, request
# import sqlite3

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/check_chellan', methods=['POST'])
# def check_chellan():
#     license_plate = request.form['license_plate']
#     chellan_details = get_chellan_details(license_plate)
#     return render_template('result.html', chellan_details=chellan_details)

# def get_chellan_details(license_plate):
#     conn = sqlite3.connect('database.db')
#     cursor = conn.cursor()

#     cursor.execute("SELECT * FROM riders WHERE LicensePlate = ?", (license_plate,))
#     rider = cursor.fetchone()

#     conn.close()

#     if rider:
#         return rider
#     else:
#         return None

# if __name__ == '__main__':
#     app.run(debug=True)


# third version starts from here 
'''
import webbrowser
import sqlite3
from flask import Flask, render_template, request

app = Flask(__name__)

def open_browser():
    # Open default web browser with the URL of the Flask server
    webbrowser.open_new('http://127.0.0.1:5000/')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_chellan', methods=['POST'])
def check_chellan():
    license_plate = request.form['license_plate']
    license_plate= license_plate.replace(" ","")
    chellan_details = get_chellan_details(license_plate)
    return render_template('result.html', chellan_details=chellan_details)

def get_chellan_details(license_plate):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM riders WHERE LicensePlate = ?", (license_plate,))
    rider = cursor.fetchone()

    conn.close()

    return rider

if __name__ == '__main__':
    open_browser()  # Open the web browser
    app.run(debug=True)
    '''


'''
import webbrowser
import sqlite3
from flask import Flask, render_template, request

app = Flask(__name__)

DATABASE = 'database.db'  # Ensure this is the correct database file


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row  # To access columns by name
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Recreate the "riders" table (only once when needed)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS riders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            LicensePlate TEXT UNIQUE,
            Name TEXT,
            ChellanAmount INTEGER
        );
    """)

    # Insert sample data only if table is empty
    cursor.execute("SELECT COUNT(*) FROM riders")
    count = cursor.fetchone()[0]
    if count == 0:
        cursor.execute("INSERT INTO riders (LicensePlate, Name, ChellanAmount) VALUES ('AP1234XYZ', 'John Doe', 500);")
        conn.commit()

    conn.close()


def open_browser():
    # Open default web browser with the URL of the Flask server
    webbrowser.open_new('http://127.0.0.1:5000/')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check_chellan', methods=['POST'])
def check_chellan():
    license_plate = request.form['license_plate'].replace(" ", "")
    chellan_details = get_chellan_details(license_plate)
    return render_template('result.html', chellan_details=chellan_details)


def get_chellan_details(license_plate):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Ensure uniform license plate format (uppercase, no spaces)
    license_plate = license_plate.replace(" ", "").upper()

    cursor.execute("SELECT * FROM riders WHERE UPPER(LicensePlate) = ?", (license_plate,))
    rider = cursor.fetchone()

    conn.close()
    
    return rider


if __name__ == '__main__':
    init_db()  # Initialize DB and table, only run once
    open_browser()  # Open the web browser
    app.run(debug=True)
'''
import sqlite3
from flask import Flask, render_template, request

app = Flask(__name__)

# Connect to the SQLite database
DATABASE = 'database.db'



def add_sample_data():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Check if table exists, create if not (no need to drop if you're appending)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS riders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        license_plate TEXT UNIQUE,
        issued_chellan INTEGER,
        name TEXT,
        phone_number TEXT
    );
    """)

    # Check if data exists to avoid duplicates (important!)
    cursor.execute("SELECT COUNT(*) FROM riders")
    count = cursor.fetchone()[0]

    if count == 0:  # Only insert if the table is empty
        data = [  # List of tuples for data insertion (REPLACE WITH YOUR DATA)
            ('AP1234XYZ', 1, 'Guru', '9123456789'),
            ('TN5678ABC', 0, 'Baba', '8741259635'),
            ('KA01AA1111', 0, 'Ramesh', '9876543210'),
            ('KA02BB2222', 1, 'Suresh', '9988776655'),
            ('TS03CC3333', 0, 'Anjali', '9765432109'),
            ('AP04DD4444', 1, 'Vivek', '9654321098'),
            ('MH05EE5555', 0, 'Priya', '9543210987'),
            ('DL06FF6666', 1, 'Rahul', '9432109876'),
            ('TN07GG7777', 0, 'Sneha', '9321098765'),
            ('KA08HH8888', 1, 'Arjun', '9210987654'),
            ('TS09II9999', 0, 'Divya', '9109876543'),
            ('AP10JJ0000', 1, 'Kiran', '9098765432'),  # Corrected license plate
            ('MH11KK1111', 0, 'Meera', '8987654321'),
            ('DL12LL2222', 1, 'Vikram', '8876543210'),
            ('TN13MM3333', 0, 'Aishwarya', '8765432109'),
            ('KA14NN4444', 1, 'Siddharth', '8654321098'),  # Corrected license plate
            ('TS15OO5555', 0, 'Deepika', '8543210987'),
            ('AP16PP6666', 1, 'Ranbir', '8432109876'),
            ('MH17QQ7777', 0, 'Kareena', '8321098765'),
            ('DL18RR8888', 1, 'Shahrukh', '8210987654'),
            ('TN19SS9999', 0, 'Priyanka', '8109876543'),
            ('KA20TT0000', 1, 'Hrithik', '7987654321'),
            ('TS21UU1111', 0, 'Katrina', '7876543210'),
            ('AP22VV2222', 1, 'Salman', '7765432109'),
            ('MH23WW3333', 0, 'Jacqueline', '7654321098'),
            ('DL24XX4444', 1, 'Akshay', '7543210987'),
            ('TN25YY5555', 0, 'Anushka', '7432109876'),
            ('KA26ZZ6666', 1, 'Saif', '7321098765'),
            ('TS27AA7777', 0, 'Kangana', '7210987654'),
            ('AP28BB8888', 1, 'Ranveer', '7109876543'),
            ('MH29CC9999', 0, 'Parineeti', '6987654321'),
            ('DL30DD0000', 1, 'Varun', '6876543210'),
            ('TN31EE1111', 0, 'Alia', '6765432109'),
            ('KA32FF2222', 1, 'Sushant', '6654321098'),
            ('TS33GG3333', 0, 'Shraddha', '6543210987'),
            ('AP34HH4444', 1, 'Rajkumar', '6432109876'),
            ('MH35II5555', 0, 'Bhumi', '6321098765'),
            ('DL36JJ6666', 1, 'Ayushmann', '6210987654'),
            ('TN37KK7777', 0, 'Yami', '6109876543'),
            ('KA38LL8888', 1, 'Vicky', '5987654321'),
            ('TS39MM9999', 0, 'Kriti', '5876543210'),
            ('AP40NN0000', 1, 'Shahid', '5765432109'),
            ('MH41OO1111', 0, 'Kiara', '5654321098'),
            ('DL42PP2222', 1, 'Sidharth', '5543210987'),
            ('TN43QQ3333', 0, 'Mrunal', '5432109876'),
            ('KA44RR4444', 1, 'Tiger', '5321098765'),
            ('TS45SS5555', 0, 'Nushrratt', '5210987654'),
            ('AP46TT6666', 1, 'Kartik', '5109876543'),
            ('MH47UU7777', 0, 'Janhvi', '4987654321'),
            ('DL48VV8888', 1, 'Aditya', '4876543210'),
            ('TN49WW9999', 0, 'Sara', '4765432109'),
            ('KA50XX0000', 1, 'Ishaan', '4654321098')
        ]

        try:
            cursor.executemany("INSERT INTO riders (license_plate, issued_chellan, name, phone_number) VALUES (?, ?, ?, ?)", data)
            conn.commit()
            print("Sample data inserted successfully!")

            # Verify data insertion (print what's in the table)
            cursor.execute("SELECT * FROM riders")
            all_data = cursor.fetchall()
            print("Data in table after insertion:", all_data)  # Check what was inserted

        except sqlite3.Error as e:
            print(f"Error during data insertion: {e}")
            conn.rollback()  # Rollback on error
    else:
        print("Data already exists. Skipping insertion.")

    conn.close()

# Call the function to insert data (you can move this to a different part of your code)
add_sample_data()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/check_chellan', methods=['POST'])
def check_chellans():
    if request.method == 'POST':
        license_plate = request.form['license_plate'].replace(" ", "").upper()  # Normalize input
        print(f"Checking challans for: {license_plate}")  # Debugging print
        chellan_details = get_chellans(license_plate)  # Fetch details

        return render_template('result.html', license_plate=license_plate, chellan_details=chellan_details)


def get_chellans(license_plate):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        # Ensure the column name is correct (`issued_challan` instead of `issued_chellan`)
        query = "SELECT issued_challan, name, phone_number FROM riders WHERE UPPER(license_plate) = ?"
        print(f"Executing query: {query} with license_plate = {license_plate}")  # Debug print
        cursor.execute(query, (license_plate,))
        result = cursor.fetchone()
        conn.close()

        print(f"Query result: {result}")  # Debug print

        if result:
            issued_challan, name, phone_number = result
            return {"issued_challan": issued_challan, "name": name, "phone_number": phone_number}
        else:
            return None  # Return None instead of "no chellans"

    except sqlite3.Error as e:
        print(f"Database error in get_chellans: {e}")
        return "error"

    finally:
        if 'conn' in locals() and conn:
            conn.close()


if __name__ == '__main__':
    app.run(debug=True)