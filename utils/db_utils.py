import pandas as pd
import numpy as np
import pyodbc
from datetime import datetime
import math
import time
import urllib
from sqlalchemy import create_engine
from tqdm import tqdm
import json
import os
from sqlalchemy.types import NVARCHAR

class DB_Utils(): # Database basic functional class: connection, read, delete, commit    
    def connect_db(self): #functional block: connection 
        retry_cnt = 0 # retry count if it error happend over 5, break retry-loop
        while retry_cnt < 5:
            try:
                self.conn = pyodbc.connect("DRIVER={ODBC Driver 17 for SQL Server};SERVER=tcp:rag-docs.database.windows.net;PORT=1433;DATABASE=rag-document-database;UID=qcells;Pwd={Asdqwe123!@#}")
                self.cursor = self.conn.cursor()
                break
            except Exception as e:
                print('{} DB CONNECTION RE-TRYING NOW COUNT: {} Times'.format(param, retry_cnt))
                retry_cnt = retry_cnt + 1
                time.sleep(5)
                pass
            if retry_cnt == 5:
                raise ConnectionError('ConnectionError: DB CONNECTION FAILED <br>'.format(param))
    
    def fetch_data(self, sql): #functional block: read
        try:
            self.connect_db()
            self.cursor.execute(sql)
            row = self.cursor.fetchall()
            row = [list(i) for i in row]
            col_names = [item[0] for item in self.cursor.description]
            self.conn.close()
            return pd.DataFrame(row, columns=col_names)
        except Exception as e:
            raise ConnectionError('ConnectionError: RPA FAILED TO FETCH DATA TO AZURE DB TABLES <br>', str(e))

    def sql_execute_commit(self, sql): #functional block: commit
        self.connect_db()
        self.cursor.execute(sql) #Error point
        self.conn.commit()
        self.conn.close()

    def insert_pd_tosql(self, tablename, df): #functional block: insert
        quoted = urllib.parse.quote_plus("DRIVER={ODBC Driver 17 for SQL Server};SERVER=tcp:rag-docs.database.windows.net;PORT=1433;DATABASE=rag-document-database;UID=qcells;Pwd={Asdqwe123!@#}")
        self.engine = create_engine('mssql+pyodbc:///?odbc_connect={}'.format(quoted),  fast_executemany=False) # try connect database with quoted
        print('[EVENT] INSERT INFO: \n', '\t DB NAME: ', 'rag-document-database', '\n' ,'\t TABLE NAME: ', tablename, '\n', '\t TABLE COLUMNS ARE BELOW', df.columns)
        df.to_sql(tablename, con=self.engine, if_exists='append', method='multi', index=False,schema="dbo", chunksize=(math.floor(2100/len(df.columns))-1), dtype = {col_name: NVARCHAR for col_name in df.columns}) # write data in tables                
        self.engine.dispose() #disconnect database engine
        del quoted, self.engine
