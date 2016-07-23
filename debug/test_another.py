from __future__ import print_function
import pdftableextract as pdf
import pprint

DEBUG = False

if DEBUG:
    import random
    import matplotlib
    matplotlib.use('AGG')
    from matplotlib.image import imsave

    def debug_imsave(name, image):
        namw="out/"+name
        print ("DEBUG: Saving {}".format(name))
        imsave(name,image)
else:
    debug_imsave=None


start_page=11
end_page=13
infile="059285.pdf"
outfilename="out/{}-059285.html"
checkall = DEBUG
out_xml=outfilename.replace("html","xml").format("xml-{}-{}".format(start_page, end_page))


def notify_page(page):
    print ("Processing page {:04d}.".format(page))

proc = pdf.Extractor(
    infile=infile,
    checkall=checkall,
    startpage=start_page,
    endpage=end_page,
    outfilename=outfilename,
    bitmap_resolution=72,
    greyscale_threshold=50,
    notify=notify_page,
    imsave=debug_imsave,
)

proc.process()

cells=proc.cells()

proc.output(table_html_filename=outfilename)

proc.xml_write(open(out_xml,'wb'))

def proc(p, check=False):
    p+=1
    outfilename="out/page-{:04d}.html".format(p)
    print ("Processing page {:04d}.".format(p))
    cells, text= pdf.process_page(infile,
                            p,
                            bitmap_resolution=72,
                            outfilename=outfilename,
                            greyscale_threshold=50,
                            checkall=check,
                            rest_text=True
    )
    pdf.output(cells, p, output_type="table_html", table_html_filename=outfilename, infile=infile, name=infile, text=text)
    return cells

#cells = [pdf.process_page("./background-checks.pdf",
#cells = [proc(p, check=check) for p in pages]

#flatten the cells structure
#cells = [item for sublist in cells for item in sublist]

#pprint.pprint (cells)



quit()
#without any options, process_page picks up a blank table at the top of the page.
#so choose table '1'
li = pdf.table_to_list(cells, pages)[1]

#li is a list of lists, the first line is the header, last is the footer (for this table only!)
#column '0' contains store names
#row '1' contains column headings
#data is row '2' through '-1'
