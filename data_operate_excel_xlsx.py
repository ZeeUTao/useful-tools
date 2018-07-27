import xlrd

xl=xlrd.open_workbook(r'example.xlsx')
sheet=xl.sheet_by_index(0)  # xlsx

# rows= sheet.row_values(n)    #contents in n th row 
# cols= sheet.col_values(n)   #contents in n th column

log = sheet.col_values(0)
log_q = sheet.col_values(1)
log_t = sheet.col_values(2)

BM = sheet.col_values(4)
BM_q = sheet.col_values(5)
BM_t = sheet.col_values(6)

# they are list form
