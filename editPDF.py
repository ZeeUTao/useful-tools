from PyPDF2 import PdfFileReader, PdfFileWriter
import numpy as np

name = r"test"
readfile = name + r".pdf"
outfile =  name + r"_out.pdf"
 
pdfReader = PdfFileReader(open(readfile, 'rb'))
pdfFileWriter = PdfFileWriter()
numPages = pdfReader.getNumPages()
#the index of first page is 0
pagelist=np.arange(5,0,-1)   

for index in pagelist:
    pageObj = pdfReader.getPage(index)
    pdfFileWriter.addPage(pageObj)
pdfFileWriter.write(open(outfile, 'wb'))

