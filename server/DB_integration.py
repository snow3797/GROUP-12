import pyodbc
from datetime import datetime

def insert_metadata(user_id, imagedirectory, openai_response, diagnosis):
    try:
        # Connection string (edit this to match your setup)
        connection_string = (
            "Driver={ODBC Driver 17 for SQL Server};"
            "Server=localhost;"
            "Database=MAIZEDISEASE;"
            "Trusted_Connection=yes;"
        )
        
        with pyodbc.connect(connection_string) as conn:
            cursor = conn.cursor()

            sql_insert = """
            INSERT INTO [dbo].[METADATA] 
                ([user_id], [TIME], [imagedirectory], [openai_response], [DIAGNOSIS])
            VALUES (?, ?, ?, ?, ?);
            """

            current_time = datetime.now()
            cursor.execute(sql_insert, (user_id, current_time, imagedirectory, openai_response, diagnosis))
            conn.commit()

        print("✅ Metadata inserted successfully")

    except pyodbc.Error as e:
        print("❌ Database error:", e)

# Example test
#if __name__ == "__main__":
 #   insert_metadata(
  #      user_id='exampleUser',
   #     imagedirectory=r'path/to/image/directory',
    #    openai_response='Sample OpenAI response',
     #   diagnosis='Sample Diagnosis'
    #)
