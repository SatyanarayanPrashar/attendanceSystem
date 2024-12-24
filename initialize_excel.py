import openpyxl
from openpyxl import Workbook

def initialize_excel():
    try:
        # Check if the file exists
        workbook = openpyxl.load_workbook("attendance.xlsx")
        print("Attendance file already exists.")
    except FileNotFoundError:
        # Create a new workbook
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Attendance"
        sheet.append(["Name", "Date", "Time"])
        workbook.save("attendance.xlsx")
        print("Attendance file created.")

if __name__ == "__main__":
    initialize_excel()
