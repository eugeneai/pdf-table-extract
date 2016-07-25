from __future__ import print_function
from nose.tools import *
import pdftableextract as pdf
import pprint
import os.path

DEBUG = False
start_page = 1
end_page = 13
infile = os.path.join(os.path.dirname(__file__),"059285.pdf")
outfilename = "out/{}-059285.html"
checkall=DEBUG

class test_examples:
    def setUp(self):
        self.proc = pdf.Extractor(infile=infile,
                                  checkall=checkall,
                                  outfilename=outfilename,
                                  bitmap_resolution=72,
                                  greyscale_threshold=50,
                                  line_length=2,
                                  )
    def tearDown(self):
        del self.proc

    @raises(ValueError)
    def test_no_pages_set(self):
        self.proc.process()

    def test_text_page(self):
        outfile="test_text_page.xml"
        o=open(outfile,"wb")
        self.proc.set_pages(1)
        self.proc.process()
        self.proc.xml_write(o)
        o.close()
