from __future__ import print_function
import pdftableextract as pdf
import pprint
import os.path

DEBUG = False
start_page = 1
end_page = 240
# start_page = 235
# end_page = 235
infile = "059285.pdf"
base, ext = os.path.splitext(infile)
outfilename = "out/{{}}-{}".format(base)

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
out_html = outfilename.format("page-{0}-{1}") + ".xhtml"
out_xml = outfilename.format("xml-{0}-{1}") + ".xml"
all_names = [out_html, out_xml]
all_names = [n.format(start_page, end_page) for n in all_names]
out_html, out_xml = all_names


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
    #line_length=0.2,
    imsave=debug_imsave, )

if __name__ == "__main__":
    proc.process()

    cells = proc.cells()

    proc.reduce(inplace=True, remove_pages=False)
    proc.xml_write(open(out_xml, 'wb'))
    proc.xhtml_write(open(out_html, "wb"))
    # proc.output(table_html_filename=outfilename)
