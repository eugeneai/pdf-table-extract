from __future__ import print_function
import pdftableextract as pdf
import pprint

DEBUG = False
start_page = 1
end_page = 12
infile = "059285.pdf"
outfilename = "out/{}-059285.html"

if DEBUG:
    import matplotlib
    matplotlib.use('AGG')
    from matplotlib.image import imsave

    def debug_imsave(name, image):
        print("DEBUG: Saving {}".format(name))
        imsave(name, image)
else:
    debug_imsave = None

checkall = DEBUG
out_xml = outfilename.replace(
    "html", "xml").format("xml-{}-{}".format(start_page, end_page))


def notify_page(page):
    print("Processing page {:04d}.".format(page), end='\r')


proc = pdf.Extractor(
    infile=infile,
    checkall=checkall,
    startpage=start_page,
    endpage=end_page,
    outfilename=outfilename,
    bitmap_resolution=72,
    greyscale_threshold=50,
    notify=notify_page,
    line_length=0.2,
    imsave=debug_imsave, )

if __name__ == "__main__":
    proc.process()

    cells = proc.cells()

    proc.xml_write(open(out_xml, 'wb'))

    proc.output(table_html_filename=outfilename)
